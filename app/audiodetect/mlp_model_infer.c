#include "mlp_model_infer.h"
#include "mlp_weights.h"
#include <math.h>
int mlp_model_predict(const float features[92],float probabilities[5])
{
  float normalized[FEATURE_DIM],hidden[HIDDEN_DIM],logits[NUM_CLASSES];
  for(int i=0;i<FEATURE_DIM;i++)normalized[i]=(features[i]-FEATURE_MEAN[i])/(FEATURE_STD[i]==0?1:FEATURE_STD[i]);
  for(int j=0;j<HIDDEN_DIM;j++){
    float v=(float)B1_INT8[j]*B1_SCALE;
    for(int i=0;i<FEATURE_DIM;i++)v+=normalized[i]*(float)W1_INT8[i][j]*W1_SCALE;
    hidden[j]=v>0?v:0;
  }
  for(int j=0;j<NUM_CLASSES;j++){
    float v=(float)B2_INT8[j]*B2_SCALE;
    for(int i=0;i<HIDDEN_DIM;i++)v+=hidden[i]*(float)W2_INT8[i][j]*W2_SCALE;
    logits[j]=v;
  }
  float maximum=logits[0],sum=0;int pred=0;
  for(int i=1;i<NUM_CLASSES;i++)if(logits[i]>maximum)maximum=logits[i];
  for(int i=0;i<NUM_CLASSES;i++){probabilities[i]=expf(logits[i]-maximum);sum+=probabilities[i];}
  for(int i=0;i<NUM_CLASSES;i++){probabilities[i]/=sum;if(probabilities[i]>probabilities[pred])pred=i;}
  return pred;
}
