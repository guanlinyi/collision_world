"""Image-only motion measurements from prepared consecutive RGB crops.

No teachers, pose, depth, past-state revision, training or network access.
Coordinates are native image pixels, not rescaled crop pixels.
"""
import cv2
import numpy as np


def crop_on_canvas(rgb, mask, bbox, bounds, scale=2):
    x0,y0,x1,y1=map(float,bbox)
    left,top,right,bottom=bounds
    width=int((right-left)*scale);height=int((bottom-top)*scale)
    # Crop RGB was bilinearly resized from [x0:x1,y0:y1].  Pixel-center
    # coordinates are continuous here; rounded ray pixels are not RGB centers.
    xx=(np.arange(width)+.5)/scale+left
    yy=(np.arange(height)+.5)/scale+top
    mapx=np.broadcast_to(((xx-x0)/(x1-x0)*rgb.shape[1]-.5)[None],(height,width)).astype(np.float32)
    mapy=np.broadcast_to(((yy-y0)/(y1-y0)*rgb.shape[0]-.5)[:,None],(height,width)).astype(np.float32)
    grey=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    image=cv2.remap(grey,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    foreground=cv2.remap(mask.astype(np.uint8)*255,mapx,mapy,cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    inside=((xx>=x0)&(xx<x1))[None]&((yy>=y0)&(yy<y1))[:,None]
    return image,foreground,inside.astype(np.uint8)


def sample_mask(mask, points):
    p=np.round(points).astype(int)
    valid=(p[:,0]>=0)&(p[:,0]<mask.shape[1])&(p[:,1]>=0)&(p[:,1]<mask.shape[0])
    result=np.zeros(len(p),bool)
    result[valid]=mask[p[valid,1],p[valid,0]]>0
    return result


def measure_normalized_pair(rgb0,mask0,bbox0,rgb1,mask1,bbox1,max_features=80):
    """Track prepared crops directly, then restore original pixel coordinates.

    Avoids one extra image reconstruction/resampling, but native information lost
    before crop preparation cannot be recovered.  Crop-relative displacement is
    never reported as native image motion.
    """
    b0,b1=np.asarray(bbox0,float),np.asarray(bbox1,float)
    wh0=b0[2:]-b0[:2];wh1=b1[2:]-b1[:2]
    a=cv2.cvtColor(rgb0,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY)
    scale=64/max(wh0)
    radius=max(1,int(np.ceil(64/min(wh0))))
    foreground=cv2.erode(mask0.astype(np.uint8)*255,np.ones((2*radius+1,2*radius+1),np.uint8))
    points=cv2.goodFeaturesToTrack(a,maxCorners=max_features,qualityLevel=.02,
                                  minDistance=max(2.,3*scale),mask=foreground,blockSize=3)
    if points is None:
        return dict(status='no_features',p0=np.empty((0,2)),p1=np.empty((0,2)),fb_error=np.empty(0),photo_error=np.empty(0),detected=0)
    p=points.astype(np.float32)
    kwargs=dict(winSize=(15,15),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01),minEigThreshold=1e-4)
    q,forward,error=cv2.calcOpticalFlowPyrLK(a,b,p,p.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**kwargs)
    back,backward,_=cv2.calcOpticalFlowPyrLK(b,a,q,p.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**kwargs)
    fb=np.linalg.norm((back.reshape(-1,2)-p.reshape(-1,2))*wh0/64,axis=1)
    photo=error.ravel()
    valid=(forward.ravel()>0)&(backward.ravel()>0)&np.isfinite(q.reshape(-1,2)).all(1)
    valid&=(fb<=1.)&(photo<=25.)&sample_mask(mask1,q.reshape(-1,2))
    native0=(p.reshape(-1,2)[valid]+.5)*wh0/64+b0[:2]-.5
    native1=(q.reshape(-1,2)[valid]+.5)*wh1/64+b1[:2]-.5
    return dict(status='measured' if valid.sum()>=4 else 'insufficient_tracks',p0=native0,p1=native1,
                fb_error=fb[valid],photo_error=photo[valid],detected=len(p))


def measure_pair(rgb0,mask0,bbox0,rgb1,mask1,bbox1,max_features=80,scale=2):
    boxes=np.array([bbox0,bbox1],float)
    bounds=(int(np.floor(boxes[:,0].min()))-12,int(np.floor(boxes[:,1].min()))-12,
            int(np.ceil(boxes[:,2].max()))+12,int(np.ceil(boxes[:,3].max()))+12)
    a,ma,ia=crop_on_canvas(rgb0,mask0,bbox0,bounds,scale)
    b,mb,ib=crop_on_canvas(rgb1,mask1,bbox1,bounds,scale)
    # A 1-native-pixel foreground guard reduces silhouette/background tracks.
    eroded=cv2.erode(ma,np.ones((2*scale+1,2*scale+1),np.uint8))
    points=cv2.goodFeaturesToTrack(a,maxCorners=max_features,qualityLevel=.02,
                                  minDistance=3*scale,mask=eroded,blockSize=3*scale)
    if points is None:
        return dict(status='no_features',p0=np.empty((0,2)),p1=np.empty((0,2)),
                    fb_error=np.empty(0),photo_error=np.empty(0),detected=0)
    p=points.astype(np.float32)
    # Bbox shift is an observable initializer, not assumed vehicle velocity.
    shift=(boxes[1,:2]+boxes[1,2:]-boxes[0,:2]-boxes[0,2:])/2*scale
    initial=p+shift.astype(np.float32)
    kwargs=dict(winSize=(15*scale+1,15*scale+1),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01),
                minEigThreshold=1e-4)
    q,forward,error=cv2.calcOpticalFlowPyrLK(a,b,p,initial,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**kwargs)
    back,backward,_=cv2.calcOpticalFlowPyrLK(b,a,q,p.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**kwargs)
    fb=np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2),axis=1)/scale
    photo=error.ravel()
    valid=(forward.ravel()>0)&(backward.ravel()>0)&np.isfinite(q.reshape(-1,2)).all(1)
    valid&=(fb<=1.)&(photo<=25.)&sample_mask(mb,q.reshape(-1,2))
    # Require centers remain within the real crop.  A crop boundary may change,
    # so this filter alone does not certify patch validity or true correspondence.
    valid&=sample_mask(ia,p.reshape(-1,2))&sample_mask(ib,q.reshape(-1,2))
    native0=(p.reshape(-1,2)[valid]+.5)/scale+np.array(bounds[:2])-.5
    native1=(q.reshape(-1,2)[valid]+.5)/scale+np.array(bounds[:2])-.5
    return dict(status='measured' if valid.sum()>=4 else 'insufficient_tracks',p0=native0,p1=native1,
                fb_error=fb[valid],photo_error=photo[valid],detected=len(p),
                native_crop_wh=np.diff(boxes[:,[0,2]],axis=1).ravel().tolist())


def project(points,q,t,k):
    camera=(np.asarray(points)-t)@q
    depth=camera[:,2]
    uv=camera[:,:2]/np.maximum(depth[:,None],1e-8)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    return uv,depth


def rigid_flow(local_points,center0,rotation0,center1,rotation1,q0,t0,k0,q1,t1,k1):
    a,d0=project(np.asarray(local_points)@rotation0.T+center0,q0,t0,k0)
    b,d1=project(np.asarray(local_points)@rotation1.T+center1,q1,t1,k1)
    return b-a,np.minimum(d0,d1)
