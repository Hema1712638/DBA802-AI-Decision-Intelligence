"""Reproducible synthetic proof of concept; NOT organization-derived evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier

OUT=Path(__file__).resolve().parent
SEED=20261019
rng=np.random.default_rng(SEED)
n=6000
features=['business_value','urgency','customer_impact','strategic_alignment','stakeholder_support','implementation_effort','delivery_risk','regulatory_criticality','dependencies','data_quality']
X=pd.DataFrame({c:rng.uniform(0,1,n) for c in features})
# Domain structure: correlated strategic and stakeholder perspectives, imperfectly observed.
X['stakeholder_support']=np.clip(.52*X['business_value']+.48*X['stakeholder_support']+rng.normal(0,.12,n),0,1)
X['data_quality']=np.clip(.55*X['strategic_alignment']+.45*X['data_quality']+rng.normal(0,.12,n),0,1)
# Simulated historical human decisions, driven by noisy, nonlinear latent business priorities.
latent=(.24*X.business_value+.16*X.urgency+.15*X.strategic_alignment+
        .13*X.customer_impact+.08*X.stakeholder_support+.11*X.regulatory_criticality-
        .13*X.implementation_effort-.08*X.delivery_risk-.04*X.dependencies+
        .10*((X.urgency>.72)&(X.regulatory_criticality>.65)).astype(float)+
        .08*X.strategic_alignment*X.data_quality+
        rng.normal(0,.115,n))
# Global quantiles fixed before splitting; label-generating thresholds are not made available to models.
cut=np.quantile(latent,[1/3,2/3]); y=np.digitize(latent,cut).astype(int)
idx=np.arange(n)
train_idx,temp_idx=train_test_split(idx,test_size=.4,random_state=SEED,stratify=y)
val_idx,test_idx=train_test_split(temp_idx,test_size=.5,random_state=SEED,stratify=y[temp_idx])
# Transparent domain heuristics, designed without looking at test labels.
def rule_prob(df):
 s=(.24*df.business_value+.19*df.urgency+.17*df.strategic_alignment+
 .13*df.customer_impact+.10*df.stakeholder_support+.13*df.regulatory_criticality-
 .14*df.implementation_effort-.09*df.delivery_risk-.06*df.dependencies+
 .09*((df.regulatory_criticality>.8)&(df.urgency>.75)).astype(float))
 # Explicit transparent policy thresholds are fixed using TRAIN feature-score quantiles only.
 return s.to_numpy()
s_train=rule_prob(X.iloc[train_idx]); bounds=np.quantile(s_train,[1/3,2/3]); temp=.10
def rule_probs(indices):
 s=rule_prob(X.iloc[indices]); centers=np.array([bounds[0]-.13,(bounds[0]+bounds[1])/2,bounds[1]+.13])
 logits=-((s[:,None]-centers[None,:])/temp)**2/2
 logits-=logits.max(axis=1,keepdims=True); z=np.exp(logits);return z/z.sum(axis=1,keepdims=True)
clf=XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.055,subsample=.85,colsample_bytree=.90,random_state=SEED,n_jobs=2,objective='multi:softprob',eval_metric='mlogloss')
clf.fit(X.iloc[train_idx],y[train_idx])
scaler=StandardScaler().fit(X.iloc[train_idx]); linear=LogisticRegression(max_iter=500,random_state=SEED)
linear.fit(scaler.transform(X.iloc[train_idx]),y[train_idx])
val_ml=clf.predict_proba(X.iloc[val_idx]); val_rule=rule_probs(val_idx)
# Only validation is used for hybrid weight selection.
weights=np.linspace(0,1,11)
score=[f1_score(y[val_idx],np.argmax((1-w)*val_ml+w*val_rule,axis=1),average='macro') for w in weights]
w=float(weights[int(np.argmax(score))]); print('Chosen rule blend weight:',w)
results=[]
pr={'Majority class':np.tile(np.bincount(y[train_idx],minlength=3).argmax(),len(test_idx)),
'Logistic regression':linear.predict(scaler.transform(X.iloc[test_idx])),
'Knowledge rules only':rule_probs(test_idx).argmax(axis=1),
'XGBoost only':clf.predict(X.iloc[test_idx]),
'Hybrid AI + rules':np.argmax((1-w)*clf.predict_proba(X.iloc[test_idx])+w*rule_probs(test_idx),axis=1)}
for name,pred in pr.items():
 results.append({'Method':name,'Macro-F1':round(float(f1_score(y[test_idx],pred,average='macro')),4),'Balanced accuracy':round(float(balanced_accuracy_score(y[test_idx],pred)),4),'Accuracy':round(float(accuracy_score(y[test_idx],pred)),4)})
res=pd.DataFrame(results);res.to_csv(OUT/'table1_metrics.csv',index=False)
# Paired bootstrap uncertainty for change in macro-F1; held-out test only.
a=pr['XGBoost only']; b=pr['Hybrid AI + rules']; ys=y[test_idx]; boot=[]
for _ in range(1000):
 ix=rng.integers(0,len(ys),len(ys));boot.append(f1_score(ys[ix],b[ix],average='macro')-f1_score(ys[ix],a[ix],average='macro'))
delta=f1_score(ys,b,average='macro')-f1_score(ys,a,average='macro')
ci=np.quantile(boot,[.025,.975]);
meta={'seed':SEED,'n':n,'train':len(train_idx),'validation':len(val_idx),'test':len(test_idx),'blend_weight':w,'test_macro_f1_delta_hybrid_minus_xgb':round(float(delta),5),'bootstrap_95pct_CI':[round(float(q),5) for q in ci], 'synthetic_data_only':True}
(OUT/'experiment_summary.json').write_text(json.dumps(meta,indent=2))
# Figure 1: Original conceptual process graphic.
fig,ax=plt.subplots(figsize=(10,3.2));ax.set_xlim(0,10);ax.set_ylim(0,3.2);ax.axis('off')
boxes=[(.1,1.2,2.1,'Synthetic project\nrequirements'),(2.75,1.2,2.1,'ML prediction\n+ stakeholder rules'),(5.4,1.2,2.1,'Validation-tuned\nhybrid scoring'),(8.05,1.2,1.8,'Priority class\n+ explanation')]
for x,y0,width,label in boxes:
 ax.add_patch(plt.Rectangle((x,y0),width,1,fc='#e7f0fc',ec='#24558a',lw=1.6));ax.text(x+width/2,y0+.5,label,ha='center',va='center',fontsize=10)
for x in [2.25,4.9,7.55]:ax.annotate('',xy=(x+.43,1.7),xytext=(x,1.7),arrowprops=dict(arrowstyle='->',lw=1.8))
ax.text(5,2.75,'Human-governed decision-support workflow (conceptual)',ha='center',weight='bold',fontsize=13)
fig.savefig(OUT/'figure1_framework.png',dpi=220,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5));xx=np.arange(len(res));ww=.25
for j,metric in enumerate(['Macro-F1','Balanced accuracy','Accuracy']):ax.bar(xx+(j-1)*ww,res[metric],ww,label=metric)
ax.set_xticks(xx,res['Method'],rotation=16,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Held-out test performance');ax.legend(ncol=3,loc='upper center');ax.grid(axis='y',alpha=.18)
fig.tight_layout();fig.savefig(OUT/'figure2_results.png',dpi=220,bbox_inches='tight');plt.close(fig)
# Class count table.
counts=pd.DataFrame({'Priority':['Low','Medium','High'],'Train':[int((y[train_idx]==k).sum()) for k in range(3)],'Validation':[int((y[val_idx]==k).sum()) for k in range(3)],'Test':[int((y[test_idx]==k).sum()) for k in range(3)]})
counts.to_csv(OUT/'table2_class_counts.csv',index=False)
# Optional SHAP summary: separate from outcome to avoid dependence on library version.
try:
 import shap
 expl=shap.TreeExplainer(clf); sv=expl.shap_values(X.iloc[test_idx[:120]])
 ar=np.asarray(sv); importance=np.mean(np.abs(ar),axis=tuple(range(ar.ndim))[1:]) if ar.ndim==2 else np.mean(np.abs(ar),axis=(0,2))
 if len(importance)==len(features):pd.DataFrame({'Feature':features,'Mean abs SHAP':importance}).sort_values('Mean abs SHAP',ascending=False).to_csv(OUT/'shap_feature_importance.csv',index=False)
except Exception as exc:
 (OUT/'shap_note.txt').write_text('SHAP optional calculation unavailable: '+str(exc))
print(res.to_string(index=False)); print(meta)
