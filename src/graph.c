#include<stdlib.h>
#include<string.h>
#include<math.h>

#include "graph.h"
#include "ops.h"

GraphContext* init_graph(int t_cap, int p_cap, int i_cap){
    GraphContext* ctx = (GraphContext*)malloc(sizeof(GraphContext));
    
    ctx->tape = (Tensor**)malloc(sizeof(Tensor*) * t_cap);
    ctx->params = (Tensor**)malloc(sizeof(Tensor*) * p_cap);
    ctx->inputs = (Tensor**)malloc(sizeof(Tensor*) * i_cap);

    ctx->t_cap = t_cap; ctx->t_count = 0;
    ctx->p_cap = p_cap; ctx->p_count = 0;
    ctx->i_cap = i_cap; ctx->i_count = 0;

    return ctx;
}

void _track_ops(GraphContext* ctx, Tensor* t){
    if(ctx->t_count >= ctx->t_cap){
        ctx->t_cap *= 2;
        ctx->tape = (Tensor**)realloc(ctx->tape, sizeof(Tensor*) * ctx->t_cap);
    }

    ctx->tape[ctx->t_count++] =  t;
}


void _track_params(GraphContext* ctx, Tensor* t){
    if(ctx->p_count >= ctx->p_cap){
        ctx->p_cap *= 2;
        ctx->params = (Tensor**)realloc(ctx->params, sizeof(Tensor*) * ctx->p_cap);
    }

    ctx->params[ctx->p_count++] =  t;
}

void _track_inputs(GraphContext* ctx, Tensor* t){
    if(ctx->i_count >= ctx->i_cap){
        ctx->i_cap *= 2;
        ctx->inputs = (Tensor**)realloc(ctx->inputs, sizeof(Tensor*) * ctx->i_cap);
    }

    ctx->inputs[ctx->i_count++] =  t;
}


Tensor* _alloc_node(GraphContext* ctx, OpType type, int size, Tensor* left, Tensor* right, int is_param){
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    
    t->size = size;
    // printf("DEBUG: Alloc Node. Addr: %p, Size Set To: %d\n", t, t->size);
    t->grad = NULL; 
    t->type = type;
    t->aux_param = 0;
    t->visited = 0;
    t->left = left;
    t->right = right;
    t->op_backward = NULL;

    if(type != OP_VAR) t->grad = (double*)calloc(size, sizeof(double));

    if(is_param){
        _track_params(ctx, t);
    } else if(type != OP_VAR){
        _track_ops(ctx, t);
    } else{
        _track_inputs(ctx, t);
    }

    return t;
}

Tensor* Input(GraphContext* ctx, double* values, int size){

    Tensor* out = _alloc_node(ctx, OP_VAR, size, NULL, NULL, 0);
    out->data = (double*)calloc(size, sizeof(double));
    out->grad = (double*)calloc(size, sizeof(double));

    if(values){
        memcpy(out->data, values, size * sizeof(double));
    }

    return out;
}

Tensor* Param(GraphContext* ctx, double* values, int size){

    Tensor* out = _alloc_node(ctx, OP_VAR, size, NULL, NULL, 1);
    out->data = (double*)calloc(size, sizeof(double));
    out->grad = (double*)calloc(size, sizeof(double));

    if(values){
        memcpy(out->data, values, size * sizeof(double));
    }

    return out;
}

Tensor* RandParam(GraphContext* ctx, int size){

    Tensor* out = _alloc_node(ctx, OP_VAR, size, NULL, NULL, 1);
    out->data = (double*)calloc(size, sizeof(double));

    out->grad = (double*)calloc(size, sizeof(double));
    double scale = sqrt(6.0 / size);
    for(int i=0; i<size; i++){
        out->data[i] = 2 * ((double) rand()/ RAND_MAX - 1.0) * scale;
    }

    return out;
}

void free_tensor(Tensor* t){
    if(!t) return;

    if(t->grad) free(t->grad);
    if(t->data) free(t->data);
    free(t);
}

void reset_tape(GraphContext* ctx){
    for(int i=0; i<ctx->t_count; i++) free_tensor(ctx->tape[i]);
    ctx->t_count = 0;
}

void reset_params(GraphContext* ctx){
    for(int i=0; i<ctx->p_count; i++) free_tensor(ctx->params[i]);
    ctx->p_count = 0;
}
void reset_inputs(GraphContext* ctx){
    for(int i=0; i<ctx->i_count; i++) free_tensor(ctx->inputs[i]);
    ctx->i_count = 0;
}

void reset_computation(GraphContext* ctx){
    reset_params(ctx);
    reset_inputs(ctx);
}

void reset_graph(GraphContext* ctx){
    if(!ctx) return;
    
    if(ctx->tape) reset_tape(ctx);
    // if(ctx->params) reset_params(ctx);
    if(ctx->inputs) reset_inputs(ctx);
}


Tensor** build_softmax(GraphContext* ctx, Tensor** logits, int num_classes){
    Tensor** probs = malloc(sizeof(Tensor*) * num_classes);
    Tensor** exp = malloc(sizeof(Tensor*) * num_classes);
    Tensor* sum_exps = NULL;

    for(int i=0; i<num_classes; i++){
        exp[i] = Exp(ctx, logits[i]);

        if(i==0){
            sum_exps = exp[i];
        } else{
            sum_exps = Add(ctx, sum_exps, exp[i]);
        }
    }

    for(int i=0; i<num_classes; i++){
        probs[i] = Div(ctx, exp[i], sum_exps);
    }

    free(exp);
    return probs;

}

Tensor* build_categorical_CE(GraphContext* ctx, Tensor** probs, Tensor** targets, int num_classes){
    Tensor* sum = NULL;

    for(int i=0; i<num_classes; i++){
        Tensor* log_p = Log(ctx, probs[i]);
        Tensor* term = Mul(ctx, targets[i], log_p);
        
        if(i==0) {sum = term;}
        else {sum = Add(ctx, sum, term);}
    }

    sum = Neg(ctx, sum);

    return sum;
}