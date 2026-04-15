#include "ops.h"
#include "graph.h"

#include<stdlib.h>
#include<math.h>
#include<stdio.h>

// backward operations (Reverse mode autodiff)
void backward_add(Tensor* a){
    Tensor* left  = a->left;
    Tensor* right = a->right;

    double *restrict g  = a->grad;
    double *restrict lg = left->grad;
    double *restrict rg = right->grad;

    int n = a->size;

    if (left->size == 1 && right->size == 1) {

        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += g[i];

        if (lg) lg[0] += sum;
        if (rg) rg[0] += sum;
    }

    else if (left->size == 1) {

        double acc  = 0.0;

        if (rg) {
            for (int i = 0; i < n; i++) {
                double gi = g[i];
                acc     += gi;     
                rg[i]   += gi;      
            }
        } else {
            for (int i = 0; i < n; i++)
                acc += g[i];
        }

        if (lg) lg[0] += acc;
    }

    else if (right->size == 1) {

        double acc  = 0.0;

        if (lg) {
            for (int i = 0; i < n; i++) {
                double gi = g[i];
                acc     += gi;
                lg[i]   += gi;
            }
        } else {
            for (int i = 0; i < n; i++)
                acc += g[i];
        }

        if (rg) rg[0] += acc;
    }

    else {

        if (lg) {
            for (int i = 0; i < n; i++)
                lg[i] += g[i];
        }
        if (rg) {
            for (int i = 0; i < n; i++)
                rg[i] += g[i];
        }
    }
}

void backward_sub(Tensor* a){
     Tensor* left  = a->left;
    Tensor* right = a->right;

    double *restrict g  = a->grad;
    double *restrict lg = left->grad;
    double *restrict rg = right->grad;

    int n = a->size;

    if (left->size == 1 && right->size == 1) {

        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += g[i];

        if (lg) lg[0] += sum;
        if (rg) rg[0] += -sum;
    }

    else if (left->size == 1) {

        double acc  = 0.0;

        if (rg) {
            for (int i = 0; i < n; i++) {
                double gi = g[i];
                acc     += gi;     
                rg[i]   += - gi;      
            }
        } else {
            for (int i = 0; i < n; i++)
                acc += g[i];
        }

        if (lg) lg[0] += acc;
    }

    else if (right->size == 1) {

        double acc  = 0.0;

        if (lg) {
            for (int i = 0; i < n; i++) {
                double gi = g[i];
                acc     += gi;
                lg[i]   += gi;
            }
        } else {
            for (int i = 0; i < n; i++)
                acc += g[i];
        }

        if (rg) rg[0] += -acc;
    }

    else {


        if (lg) {
            for (int i = 0; i < n; i++)
                lg[i] += g[i];
        }
        
        if (rg) {
            for (int i = 0; i < n; i++)
                rg[i] += -g[i];
        }
    }
}

void backward_mul(Tensor* a) {
    Tensor* left  = a->left;
    Tensor* right = a->right;

    double *restrict g  = a->grad;
    double *restrict ld = left->data;
    double *restrict rd = right->data;
    double *restrict lg = left->grad;
    double *restrict rg = right->grad;

    int n = a->size;

    if (left->size == 1 && right->size == 1) {

        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += g[i];

        if (lg) lg[0] += sum * rd[0];
        if (rg) rg[0] += sum * ld[0];
    }

    else if (left->size == 1) {

        double acc  = 0.0;
        double lval = ld[0];

        if (rg) {
            for (int i = 0; i < n; i++) {
                double gi = g[i];
                acc     += gi * rd[i];     
                rg[i]   += gi * lval;      
            }
        } else {
            for (int i = 0; i < n; i++)
                acc += g[i] * rd[i];
        }

        if (lg) lg[0] += acc;
    }

    else if (right->size == 1) {

        double acc  = 0.0;
        double rval = rd[0];

        if (lg) {
            for (int i = 0; i < n; i++) {
                double gi = g[i];
                acc     += gi * ld[i];
                lg[i]   += gi * rval;
            }
        } else {
            for (int i = 0; i < n; i++)
                acc += g[i] * ld[i];
        }

        if (rg) rg[0] += acc;
    }

    else {

        if (lg && rg) {
            for (int i = 0; i < n; i++) {
                double gi = g[i];
                lg[i] += gi * rd[i];
                rg[i] += gi * ld[i];
            }
        }
        else if (lg) {
            for (int i = 0; i < n; i++)
                lg[i] += g[i] * rd[i];
        }
        else if (rg) {
            for (int i = 0; i < n; i++)
                rg[i] += g[i] * ld[i];
        }
    }
}

void backward_square(Tensor* a){
    Tensor* x = a->left;
    for(int i=0; i<a->size; i++){
        double g = a->grad[i];
        
        if(x->grad) x->grad[i] += (2.0 * x->data[i]) * g;
    }
}


void backward_exp(Tensor* a){
    Tensor* x = a->left;
    for(int i=0; i<a->size; i++){
        
        double g = a->grad[i];
        if(x->grad) x->grad[i] += (a->data[i]) * g;
    }
}

void backward_div(Tensor* a){
    Tensor* num = a->left;
    Tensor* denom = a->right;   


    double *restrict g = a->grad;
    double *restrict ld = num->data;
    double *restrict lg = num->grad;
    double *restrict rd = denom->data;
    double *restrict rg = denom->grad;

    int n = a->size;
    const double eps = 1e-8;

    if(num->size == 1 && denom->size == 1){
        double sum = 0.0;
        
        for(int i=0; i<n; i++) sum += g[i];

        double d = rd[0];
        double d_safe = fabs(d) < eps ? (d>=0 ? eps : -eps) : d;
        
        if(lg) lg[0] += sum * (1.0 / d_safe); 
        if(rg) rg[0] += sum * ( -ld[0] / (d_safe * d_safe)); 

    } else if(num->size == 1){
        
        double acc = 0.0;
        double nval = ld[0];

        if(rg){
            for(int i=0; i<n; i++){
                
                double d = rd[i];
                double d_safe = fabs(d) < eps ? (d>=0 ? eps : -eps) : d;
                
                acc += g[i] * (1.0 / d_safe);
                rg[i] += g[i] * (-nval / (d_safe * d_safe));
    
            }

        } else{
            for(int i=0; i<n; i++){
                
                double d = rd[i];
                double d_safe = fabs(d) < eps ? (d>=0 ? eps : -eps) : d;
                
                acc += g[i] * (1.0 / d_safe);
            }
        }
        
        if(lg) lg[0] += acc;

    } else if(denom->size == 1){
        
        double acc = 0.0;
        double rval = rd[0];

        if(lg){

            double d_safe = fabs(rval) < eps ? (rval>=0 ? eps : -eps) : rval;

            for(int i=0; i<n; i++){
                
                acc += g[i] * (-ld[i] / (d_safe * d_safe));
                lg[i] += g[i] * (1.0 / d_safe);
    
            }

        } else{

            double d_safe = fabs(rval) < eps ? (rval>=0 ? eps : -eps) : rval;
                        
            for(int i=0; i<n; i++){    
                acc += g[i] * (-ld[i] / (d_safe * d_safe));
            }
        }
        
        if(rg) rg[0] += acc;

    } else{
        
        if(lg && rg){
            for(int i=0; i<n; i++){

                double d = rd[i];
                double d_safe = (fabs(d) < eps) ? (d>=0 ? eps: -eps) : d;
                lg[i] += (1.0 / d_safe) * g[i];
                rg[i] += -(ld[i] / (d_safe * d_safe)) * g[i];
            }
        } else if(lg){
                for(int i=0; i<n; i++){

                double d = rd[i];
                double d_safe = (fabs(d) < eps) ? (d>=0 ? eps: -eps) : d;
                lg[i] += (1.0 / d_safe) * g[i];
            }
            
        } else if(rg){
                for(int i=0; i<n; i++){

                double d = rd[i];
                double d_safe = (fabs(d) < eps) ? (d>=0 ? eps: -eps) : d;
                rg[i] += g[i] * ( -ld[i] / (d_safe * d_safe) );
            }
            
        }
    }
}

void backward_neg(Tensor* a){
   
    Tensor* x = a->left;
    
    double *restrict g = a->grad;
    double *restrict  lg = x->grad;

    if(!g || !lg) return;
 
    for(int i=0; i<a->size; i++){
            lg[i] += -g[i];
    }
}

void backward_log(Tensor* a){
    Tensor* x = a->left;

    const double eps = 1e-8; // Prevent div by zero
    for(int i=0; i<a->size; i++){
        double g = a->grad[i];
        double val = x->data[i];
        // d(ln x)/dx = 1/x
        if(x->grad) x->grad[i] += g * (1.0 / (val + eps)); 
    }
}

void backward_mean(Tensor* a){

    Tensor* x = a->left;

    double g = (a->grad) ? a->grad[0] : 1.0;
    
    double *restrict lg = x->grad;

    double scale = g / x->size;
    if(lg){
        for(int i=0; i<x->size; i++){
            lg[i] += scale;
        }
    }
}

void backward_relu(Tensor* a){

    Tensor* x = a->left;

    double *restrict g = a->grad;
    double *restrict lg = x->grad;
    double *restrict ld = x->data;

    if(lg){

        for(int i=0; i<a->size; i++){
            lg[i] += (ld[i] > 0 ? 1.0 : 0.0) * g[i];
        }
    }
}

void backward_tanh(Tensor* a){
    Tensor* x = a->left;

    for(int i=0; i<a->size; i++){
        double y = a->data[i];
        double g = a->grad[i];
        if(x->grad) x->grad[i] += (1.0 - y*y) * g; // derivative of tanh
    } 

}
void backward_sigmoid(Tensor* a){
    Tensor* x = a->left;

    for(int i=0; i<a->size; i++){
    
        double g = a->grad[i];
        double y = a->data[i];
        if(x->grad) x->grad[i] += (y * (1.0 - y)) * g; // derivative of sigmoid
    } 
}

void backward_softmax(Tensor* a){
    
    Tensor* input = a->left;
    
    int sample_width = a->aux_param;
    
    if(sample_width == 0) sample_width = a->size;
    
    int total_samples = a->size / sample_width;
    
    for(int n=0; n<total_samples; n++){
        
        int start_index = n * sample_width;
        double dot = 0.0;
        
        // dz = S X (G - Dot)
        // dot
        for(int i=0; i < sample_width; i++){

            double g_i = a->grad[start_index + i];
            double s_i = a->data[start_index + i];
            dot += s_i * g_i;
        }

        for(int i=0; i < sample_width; i++){
            double s_i = a->data[start_index + i];
            double g_i = a->grad[start_index + i];
            
            if(input->grad) input->grad[start_index + i] += s_i * (g_i - dot);
        }
    }
}


// loss is scaler so every input is scaler

void backward_mse(Tensor* a){
    Tensor* pred = a->left;    
    Tensor* target = a->right;
    
    double grad_incoming = (a->grad)? a->grad[0] : 1.0;
    int n = pred->size;

    double scale = (2.0 / n) * grad_incoming;

    for(int i=0; i<pred->size; i++){
        double s = pred->data[i];
        double y = target->data[i];
        
        if(pred->grad) pred->grad[i] += scale * (s - y);
    }

}

void backward_rmse(Tensor* a){
    
    Tensor* pred = a->left;    
    Tensor* target = a->right;
    
    double grad_incoming = (a->grad)? a->grad[0] : 1.0;
    int n = pred->size;

    double rmse_val = a->data[0]; 

    if(rmse_val < 1e-8) rmse_val = 1e-8;
    
    double scale = (1.0 / n) * (1.0 / rmse_val) * grad_incoming;

    for(int i=0; i<pred->size; i++){

        double s = pred->data[i];
        double y = target->data[i];
        
        if(pred->grad) pred->grad[i] += scale * (s - y);
    }
}
void backward_mae(Tensor* a){

    Tensor* pred = a->left;    
    Tensor* target = a->right;

    int n = pred->size;
    
    double grad_incoming = (a->grad)? a->grad[0] : 1.0;
    double scale = (1.0 / n) * grad_incoming;

    for(int i=0; i<pred->size; i++){

        double s = pred->data[i];
        double y = target->data[i];
        double diff = s - y;
        double sign = 0.0;

        if(diff > 0) {sign = 1.0;}
        else if(diff < 0) {sign = -1.0;}
        else {sign = 0.0;}

        if(pred->grad) pred->grad[i] += scale * sign;
    }
}

void backward_softmax_CE(Tensor* a){
    
    Tensor* pred = a->left;
    Tensor* target = a->right;

    Tensor* logits = pred->left;
    
    if(!logits || !logits->grad) return;

    int n = pred->size;
    double grad_incoming = (a->grad) ? a->grad[0] : 1.0;
    
    double scale = (1.0 / n) * grad_incoming; // divided by mean to keep gradient safe from exploding

    for(int i=0; i<n; i++){
        double s = pred->data[i];
        double y = target->data[i];
        
        logits->grad[i] += scale * (s - y);
    }
    
}

void backward_CE(Tensor* a){
    Tensor* pred = a->left;
    Tensor* target = a->right;


    double grad_incoming = a->grad ? a->grad[0] : 1.0;
    int n = pred->size;
    
    double scale = (1.0 / n) * grad_incoming;

    double *restrict ld = pred->data;
    double *restrict  lg= pred->grad;
    double *restrict  rd= target->data;
    

    if(lg){
        for(int i=0; i<n; i++) lg[i] += scale * -( rd[i] / ld[i]);
    }
}

void backward_sparse_softmax_CE(Tensor* a){
    
    Tensor* pred = a->left;
    Tensor* target = a->right;

    Tensor* logits = pred->left;
    
    if(!logits || !logits->grad) return;

    int n = pred->size;
    double grad_incoming = (a->grad) ? a->grad[0] : 1.0;
    
    double scale = (1.0 / n) * grad_incoming;

    int corrected_class = (int)target->data[0];

    for(int i=0; i<n; i++){
        double s = pred->data[i];
        
        double y = (i == corrected_class)? 1.0 : 0.0;

        logits->grad[i] += scale * (s - y);
    }
    
}


void backward_binary_crossentropy(Tensor* a){
    
    Tensor* pred = a->left;
    Tensor* target = a->right;
    
    int n = pred->size;
    double grad_incoming = (a->grad)? a->grad[0] : 1.0;

    const double epsilon = 1e-7;
    
    double scale = (1.0 / n) * grad_incoming;    
    for(int i=0; i<pred->size; i++){
        
        double s = pred->data[i];
        double y = target->data[i];
        double s_safe = (s < epsilon) ? epsilon : ((s > 1 - epsilon) ? 1-epsilon : s);

        if(pred->grad) pred->grad[i] += scale * ((s_safe - y) / (s_safe * (1 - s_safe)));
    
    }

}

void backward_binary_crossentropy_sigmoid(Tensor* a){
    
    Tensor* pred = a->left;
    Tensor* target = a->right;
    
    Tensor* logits = pred->left;
    

    int n = pred->size;
    double grad_incoming = (a->grad)? a->grad[0] : 1.0;
    double scale = (1.0 / n) * grad_incoming;    

    
    for(int i=0; i<pred->size; i++){
        double s = pred->data[i];
        double y = target->data[i];
        logits->grad[i] += scale * (s - y);
    
    }

}

//Operations

Tensor* Add(GraphContext* ctx, Tensor* a, Tensor* b){
    // CHANGE THIS LINE:
    if(a->size != b->size) { 
        printf("Error: Size mismatch Add. A: %d, B: %d\n", a->size, b->size); 
        return NULL; 
    }
    Tensor* out = _alloc_node(ctx, OP_ADD, a->size, a, b, 0);
    out->op_backward = backward_add;

    out->data = (double *)malloc(a->size * sizeof(double));
    for(int i=0; i<a->size;i++){
        out->data[i] = a->data[i] + b->data[i];
    }

    return out;
    
}

Tensor* Sub(GraphContext* ctx, Tensor* a, Tensor* b){
    
    if(a->size != b->size) { printf("Error: Size mismatch Add\n"); return NULL; }

    Tensor* out = _alloc_node(ctx, OP_SUB, a->size, a, b, 0);
    out->op_backward = backward_sub;

    out->data = (double *)malloc(a->size * sizeof(double));
    for(int i=0; i<a->size;i++){
        out->data[i] = a->data[i] - b->data[i];
    }

    return out;
    
}

Tensor* Mul(GraphContext* ctx, Tensor* a, Tensor* b){
    
    int valid = (a->size == b->size) || (a->size == 1) || (b->size == 1);

    if(!valid) { 
        printf("Error: Size mismatch Mul. A: %d, B: %d\n", a->size, b->size); 
        return NULL; 
    }

    int out_size = (a->size > b->size) ? a->size : b->size;
    
    Tensor* out = _alloc_node(ctx, OP_MUL, out_size, a, b, 0);
    out->op_backward = backward_mul;

    out->data = (double *)malloc(out_size * sizeof(double));
    

    double *restrict ad = a->data;
    double *restrict bd = b->data;
    double *restrict od = out->data;

    // this looks like if else branch but this will save the loop to have inner loop conditional overhead, and also this is SIMD friendly
    if(a->size == 1 && b->size ==1){
        double av = ad[0]; 
        double bv = bd[0]; 
        
        for(int i=0; i<out_size; i++) od[i] = av * bv;
        
    } else if(a->size == 1){
    
        double av = a->data[0];
        for (int i=0; i<out_size; i++) od[i] = av * bd[i];
    
    } else if(b->size == 1){
        double bv = b->data[0];
        for (int i=0; i<out_size; i++) od[i] = bv * ad[i];
    } else{
        for(int i=0; i<out_size; i++) od[i] = ad[i] * bd[i];
    }

    return out;
    
}

Tensor* Square(GraphContext* ctx, Tensor* a){
    
    Tensor* out = _alloc_node(ctx, OP_SQUARE, a->size, a, NULL, 0);
    out->op_backward = backward_square;

    out->data = (double *)malloc(a->size * sizeof(double));
    for(int i=0; i<a->size;i++){
        out->data[i] = a->data[i] * a->data[i];
    }

    return out;
    
}

Tensor* Exp(GraphContext* ctx, Tensor* a){
    
    Tensor* out = _alloc_node(ctx, OP_EXP, a->size, a, NULL, 0);
    out->op_backward = backward_exp;

    out->data = (double *)malloc(a->size * sizeof(double));
    
    for(int i=0; i<a->size;i++) {
        
        double x = (a->data[i] > 88.0) ? 88.0 : (a->data[i] < -88.0 ? - 88.0 : a->data[i]);

        out->data[i] = exp(x);
    }

    return out;
    
}

Tensor* Div(GraphContext* ctx, Tensor* a, Tensor* b){
    
    Tensor* out = _alloc_node(ctx, OP_DIV, a->size, a, b, 0);
    out->op_backward = backward_div;

    out->data = (double *)malloc(a->size * sizeof(double));
    const double eps = 1e-8;

    for(int i=0; i<a->size;i++) {
        
        double num = a->data[i]; 
        double denom = b->data[i]; 
        
        double d_safe = (fabs(denom) < eps)? (denom >=0 ? eps : -eps) : denom;
        out->data[i] = (num)/(d_safe);
    }

    return out;   
}

Tensor* Neg(GraphContext* ctx, Tensor* a){
    Tensor* out = _alloc_node(ctx, OP_NEG, a->size, a, NULL, 0);
    out->op_backward = backward_neg;
    out->data = (double*)malloc(a->size * sizeof(double));

    for(int i=0; i<a->size; i++) out->data[i] = -1 * a->data[i];
    
    return out;
}


Tensor* Log(GraphContext* ctx, Tensor* a){
    Tensor* out = _alloc_node(ctx, OP_LOG, a->size, a, NULL, 0);
    out->op_backward = backward_log;
    out->data = (double*)malloc(a->size * sizeof(double));
    
    const double eps = 1e-8;
    for(int i=0; i<a->size; i++) {
        double val = a->data[i];
        if(val < eps) val = eps; // Numerical stability
        out->data[i] = log(val);
    }
    return out;
}

Tensor* Mean(GraphContext* ctx, Tensor* a){
    Tensor* out = _alloc_node(ctx, OP_MEAN, 1, a, NULL, 0);
    
    int size = a->size;
    double sum = 0.0;

    out->op_backward = backward_mean;
    out->data = (double*)malloc(1 * sizeof(double));

    for(int i=0; i<size; i++) sum += a->data[i];
    
    out->data[0] = sum / size;
    return out;
}


//Activations

Tensor* Relu(GraphContext* ctx, Tensor* a){
    
    Tensor* out = _alloc_node(ctx, OP_RELU, a->size, a, NULL, 0);
    out->data = (double*)malloc(a->size * sizeof(double));

    for(int i=0; i<a->size; i++){
        double val = a->data[i] > 0 ? a->data[i] : 0.0;
        out->data[i] = val;
    }
    out->op_backward = backward_relu;
    
    return out;
}

Tensor* Tanh(GraphContext* ctx, Tensor* a){
    
    Tensor* out = _alloc_node(ctx, OP_TANH, a->size, a, NULL, 0);
    out->data = (double*)malloc(a->size * sizeof(double));

    for(int i=0; i<a->size; i++){
        
        double z = a->data[i];
        double s = (exp(2* z) - 1.0) / (exp(2.0 * z) + 1.0);
        
        out->data[i] = s;
    }
    out->op_backward = backward_tanh;
    
    return out;
}

Tensor* Sigmoid(GraphContext* ctx, Tensor* a){
    
    Tensor* out = _alloc_node(ctx, OP_SIGMOID, a->size, a, NULL, 0);
    out->data = (double*)malloc(a->size * sizeof(double));

    for(int i=0; i<a->size; i++){
        
        double z = a->data[i];
        double s =  1.0 / (1.0 + exp(-z));
        
        out->data[i] = s;
    }
    out->op_backward = backward_sigmoid;
    
    return out;
}

Tensor* Softmax(GraphContext* ctx, Tensor* a, int sample_width){
    
    Tensor* out = _alloc_node(ctx, OP_SOFTMAX, a->size, a, NULL, 0);
    out->data = (double*)malloc(a->size * sizeof(double));
    out->aux_param = sample_width;

    int total_samples = a->size / sample_width;
    
    for(int sample_idx=0; sample_idx < total_samples; sample_idx++){
        
        
        int start_index = sample_idx * sample_width;
        double sum = 0.0;
        double max = a->data[start_index];
        
        for(int j=0; j < sample_width; j++){
            if(a->data[start_index + j] >= max) max = a->data[start_index + j];
        }

        for(int j=0; j < sample_width; j++){
            out->data[start_index + j] = exp(a->data[start_index + j] - max);
            sum += out->data[start_index + j]; 
        }

        for(int j=0; j < sample_width; j++) out->data[start_index + j] /= sum;
    }

    out->op_backward = backward_softmax;

    return out;
}

//Loss funcs

Tensor* Mse(GraphContext* ctx,Tensor* y, Tensor* y_hat){
    
    Tensor* out = _alloc_node(ctx, OP_MSE, 1, y_hat, y, 0);
    out->data = (double*)malloc(sizeof(double));

    double sum = 0.0;
    for(int i=0; i<y_hat->size; i++){
        double diff = (y_hat->data[i] - y->data[i]);
        sum += diff * diff;
    }

    out->data[0] = sum / y_hat->size;
    out->op_backward = backward_mse;
    
    return out;
}

Tensor* Rmse(GraphContext* ctx,Tensor* y, Tensor* y_hat){

    Tensor* out = _alloc_node(ctx, OP_RMSE, 1, y_hat, y, 0);
    out->data = (double*)malloc(y_hat->size * sizeof(double));

    double sum = 0.0;
    for(int i=0; i<y_hat->size; i++){
        double diff = (y_hat->data[i] - y->data[i]);
        sum += diff * diff;
    }

    out->data[0] = sqrt(sum / y->size);
    out->op_backward = backward_rmse;
    
    return out;
}

Tensor* Mae(GraphContext* ctx,Tensor* y, Tensor* y_hat){
    
    Tensor* out = _alloc_node(ctx, OP_MAE, 1, y_hat, y, 0);
    out->data = (double*)malloc(y_hat->size * sizeof(double));

    double sum = 0.0;
    for(int i=0; i<y_hat->size; i++){
        double diff = (y_hat->data[i] - y->data[i]);
        sum += (diff > 0? diff : -diff);
    }

    out->data[0] = sum / y->size;
    out->op_backward = backward_mae;
    
    return out;
}

Tensor* Softmax_CE(GraphContext* ctx, Tensor* target, Tensor* pred){

    Tensor* out = _alloc_node(ctx, OP_SOFTMAX_CE, 1, pred, target, 0);
    out->data = (double*)malloc(sizeof(double));
    
    out->op_backward = backward_softmax_CE;
    
    double loss = 0.0;
    for(int i=0; i < target->size; i++){
        
        double p = pred->data[i];
        double y = target->data[i];
        
        double eps = 1e-8;
        
        loss -= y * log(p + eps);
    }

    out->data[0] = loss;
    
    return out;
}
Tensor* Binary_CE(GraphContext* ctx, Tensor* target, Tensor* pred){
     
    Tensor* out = _alloc_node(ctx, OP_VAR, 1, pred, target, 0);
    out->data = (double*)malloc(sizeof(double));
    
    out->op_backward = backward_binary_crossentropy;
    
    double loss = 0.0;
    for(int i=0; i < target->size; i++){
        
        double p = pred->data[i];
        double y = target->data[i];
        
        double eps = 1e-7;
        
        double p_safe = (p < eps) ? eps : ((p > 1 - eps) ? 1.0 - eps : p);

        loss -= (y * log(p_safe) + (1.0 - y) * log(1.0 - p_safe));
    }

    out->data[0] = loss/target->size;
    
    return out;
}