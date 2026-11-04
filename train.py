"""Reproduce frozen artifacts. Training is offline; serving uses exported numeric models."""
from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_score, recall_score, fbeta_score, brier_score_loss
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parent
COLUMNS=['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
KEYS=['type','air','process','speed','torque','wear']
FEATURES=['Air temperature','Process temperature','Rotational speed','Torque','Tool wear','Temperature gap','Mechanical power','Wear × torque','Type H','Type L','Type M']

def features(raw, medians):
    a=np.array([[None if pd.isna(v) else float(v) for v in r[1:]] for r in raw],dtype=float)
    a=np.where(np.isnan(a),medians,a)
    air,process,speed,torque,wear=a.T
    return np.column_stack([a,process-air,speed*torque*(2*np.pi/60),wear*torque,*[(raw[:,0]==t).astype(float) for t in ['H','L','M']]])

def metrics(y,p,t):
    pred=p>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),positives=int(sum(y)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),f2=float(fbeta_score(y,pred,beta=2,zero_division=0)),accuracy=float(np.mean(pred==y)),averagePrecision=float(average_precision_score(y,p)),rocAuc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)))

def fit_models(raw,y,train):
    med=np.nanmedian(raw[train,1:].astype(float),axis=0)
    x=features(raw,med)
    scale=StandardScaler().fit(x[train])
    lr=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42).fit(scale.transform(x[train]),y[train])
    rf=RandomForestClassifier(n_estimators=96,max_depth=9,min_samples_leaf=3,class_weight='balanced_subsample',random_state=42,n_jobs=1).fit(x[train],y[train])
    return med,x,scale,lr,rf

def main():
    raw_bytes=(ROOT/'ai4i2020.csv').read_bytes();df=pd.read_csv(ROOT/'ai4i2020.csv');raw=df[COLUMNS].to_numpy();y=df['Machine failure'].to_numpy()
    idx=np.arange(len(y));tr,other=train_test_split(idx,test_size=.4,stratify=y,random_state=42);va,te=train_test_split(other,test_size=.5,stratify=y[other],random_state=42)
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    med,x,scale,lr,rf=fit_models(raw,y,tr)
    models={'lr-v1':lr,'rf-v1':rf};preds={k:m.predict_proba(scale.transform(x) if k=='lr-v1' else x)[:,1] for k,m in models.items()}
    # Select model and threshold using validation data only, then inspect held-out test once.
    selected=max(models,key=lambda k:average_precision_score(y[va],preds[k][va]))
    grid=np.arange(.05,.951,.01);threshold=float(max(grid,key=lambda t:(fbeta_score(y[va],preds[selected][va]>=t,beta=2,zero_division=0),t)))
    report={};export={}
    for k,m in models.items():
        p=preds[k]
        imp=permutation_importance(m,scale.transform(x[va]) if k=='lr-v1' else x[va],y[va],n_repeats=3,random_state=42,scoring='average_precision',n_jobs=1)
        report[k]={'name':'Logistic regression' if k=='lr-v1' else 'Random forest','version':k,'validation':metrics(y[va],p[va],threshold),'test':metrics(y[te],p[te],threshold),'importance':[{'feature':f,'drop':float(v),'std':float(s)} for f,v,s in zip(FEATURES,imp.importances_mean,imp.importances_std)],'parameters':{'class_weight':'balanced','max_iter':2000} if k=='lr-v1' else {'n_estimators':96,'max_depth':9,'min_samples_leaf':3,'class_weight':'balanced_subsample'}}
        if k=='lr-v1':export[k]={'weights':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist()}
        else:
            trees=[]
            for tree in rf.estimators_:
                t=tree.tree_;v=t.value[:,0,:];prob=v[:,1]/v.sum(axis=1)
                trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':prob.tolist()})
            export[k]={'trees':trees}
    chronological={};ct=np.arange(6000);cv=np.arange(6000,8000);ce=np.arange(8000,10000)
    cm,cx,cs,cl,cr=fit_models(raw,y,ct)
    for k,m in [('lr-v1',cl),('rf-v1',cr)]:
        cp=m.predict_proba(cs.transform(cx) if k=='lr-v1' else cx)[:,1];best=float(max(grid,key=lambda t:(fbeta_score(y[cv],cp[cv]>=t,beta=2,zero_division=0),t)))
        chronological[k]={'threshold':best,'validation':metrics(y[cv],cp[cv],best),'test':metrics(y[ce],cp[ce],best)}
    def row(i):return {k:(str(v) if k=='type' else float(v)) for k,v in zip(KEYS,raw[i])}
    order=sorted(te,key=lambda i:preds[selected][i]);sample_ids=[order[0],order[len(order)//2],order[-1]]
    ranges={key:{'min':float(np.min(raw[tr,j].astype(float))),'max':float(np.max(raw[tr,j].astype(float))),'median':float(med[j-1])} for j,key in enumerate(KEYS) if j}
    bundle={'version':'1.0.0','seed':42,'selected':selected,'threshold':round(threshold,2),'featureNames':FEATURES,'medians':med.tolist(),'ranges':ranges,'models':export,'report':report,'chronological':chronological,'dataset':{'name':'UCI AI4I 2020','synthetic':True,'doi':'10.24432/C5HS5C','license':'CC BY 4.0','sha256':hashlib.sha256(raw_bytes).hexdigest(),'rows':len(y),'positives':int(sum(y)),'split':{'train':len(tr),'validation':len(va),'test':len(te)},'splitPositives':{'train':int(sum(y[tr])),'validation':int(sum(y[va])),'test':int(sum(y[te]))}},'runtime':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},'samples':[{'label':n,'row':row(i),'actual':int(y[i]),'datasetRow':int(i+1)} for n,i in zip(['Low-score benchmark row','Mid-score benchmark row','High-score benchmark row'],sample_ids)],'holdout':[{'actual':int(y[i]),'scores':{k:float(preds[k][i]) for k in models}} for i in te],'parity':[{'row':row(i),'scores':{k:float(preds[k][i]) for k in models}} for i in te[:40]]}
    (ROOT/'model-bundle.json').write_text(json.dumps(bundle,separators=(',',':'),allow_nan=False))
    (ROOT/'split-indices.json').write_text(json.dumps({'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()}))
    print(json.dumps({'selected':selected,'threshold':round(threshold,2),'metrics':{k:r['test'] for k,r in report.items()},'chronological':chronological},indent=2))

if __name__=='__main__':main()
