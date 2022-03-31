import numpy as np

from utils.loss import *
class ComputeLossV1:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        #这里增加了一个loss function,用来计算theta的损失
        BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            #同上若启用focalloss则也需要增加theta相应的损失函数
            BCEcls, BCEobj, BCEtheta = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCEtheta, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.BCEtheta, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, BCEtheta, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        # p.shape(nl,bs,na,ny,nx,nc+6)nl为层数3，bs是batchsize,na是每层anchor的数量，ny,nx为feature map的高和宽，nc+6包含预测的(x,y,w,h,theta,obj,nc个参数用来预测类别)
        # targets (image,cat, x,y,w,h,theta)
        device = targets.device
        lcls, lbox, lobj, ltheta = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, ttheta, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                #这里获取预测的theta值，它处在第5个位置上，索引为4
                ptheta = ps[:, 4].sigmoid() * np.pi/2
                ptheta[ptheta<0.001]=np.pi/2 #这里将过小的角度直接提升为pi/2,因此预测的theta为(0.001,pi/2]

                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification and theta regression
                if self.nc > 1:  # cls loss (only if multiple classes)
                    #在求class的loss的过程中，将原版的5改为6，这里由于前面增加了一个量用来预测theta
                    t = torch.full_like(ps[:, 6:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:,6:], t)  # BCE
                    #求出theta的regression loss 并且利用pi/2进行normalize
                    ltheta +=  (ptheta - ttheta[i]).abs().mean()*2/pi
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            #这里将原版的4改为5，因为原版的第四个值用来预测theta
            obji = self.BCEobj(pi[..., 5], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        ltheta *= self.hyp['theta']

        bs = tobj.shape[0]  # batch size
        #todo 这里暂时先不讲theta的loss放入到loss计算中,先将hbb的检测在train和val过程都进行校准
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p,
                      targets):  # this function generates positive anchor targets' positions on the feature maps with different size
        # Build targets for compute_loss(), input targets(image,cat ,x,y,w,h,theta)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, ttheta, indices, anch = [], [], [], [], []

        gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain 这里gain的维度从原版的7提升到8,因为targets增加了theta量，但该量不需要进行尺度变换
        #为每个target增加一个量，该量用来记录anchor的索引
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        #增加后的targets变量从原先的(image,class,x,y,w,h,theta)变为(image,class,x,y,w,h,theta,ai),其数量变为原先的na倍
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]

            #如下两行代码获得特征向量p在层i的尺寸，利用该尺寸将normalize的target进行尺度变换
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # Match targets to anchors
            t = targets * gain
            #以下的筛选过程和theta，class等无关，主要是利用target和anchor之间的长以及宽的对应比例，来筛选出每个target符合比例范围的anchor，
            #一个target可以有多个anchor符合，也可以一个都没有，这个有点奇怪，之后需要再确认一下这部分的代码
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare

                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            theta = t[:,6] #theta 这里将theta单独从target中取出，为之后的loss计算做准备

            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append 这里从anchor的index的位置从原版的6改为7
            a = t[:, 7].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            ttheta.append(theta) #theta for the rotated boxes

        return tcls, tbox, ttheta, indices, anch
        # tcls is the categories,
        # tbox is gtbox与三个负责预测的网格的xy坐标偏移量，gtbox的宽高,
        # indices (b表示当前gtbox属于该batch内第几张图片，a表示gtbox与anchors的对应关系，负责预测的网格纵坐标，负责预测的网格横坐标)
        # anch (最匹配的anchors)