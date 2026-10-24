#ifndef MLP_MODEL_INFER_H
#define MLP_MODEL_INFER_H
/*92-dimensional front-end and five labels retain the historical order. */
int mlp_model_predict(const float features[92],float probabilities[5]);
#endif
