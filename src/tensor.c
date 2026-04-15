#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "graph.h"
#include "stensor.h"


void _build_topo(Tensor* v, Tensor*** sorted, int* idx, int* cap){
    if (v->visited) return;

    v->visited = 1;

    if(v->left) _build_topo(v->left, sorted, idx, cap);
    if(v->right) _build_topo(v->right, sorted, idx, cap);

    if(*idx >= *cap){
        *cap *= 2;
        *sorted = (Tensor**)realloc(*sorted, sizeof(Tensor*) * (*cap));
    }

    (*sorted)[(*idx)++] = v;
}


void backward(GraphContext* ctx, Tensor* root){
    
    for(int i=0; i < ctx->t_count; i++) ctx->tape[i]->visited = 0; 
    for(int i=0; i < ctx->i_count; i++) ctx->inputs[i]->visited = 0; 
    for(int i=0; i < ctx->p_count; i++) ctx->params[i]->visited = 0; 

    int max_nodes = ctx->i_cap + ctx->p_cap + ctx->t_cap;

    Tensor** sorted = (Tensor**)malloc(max_nodes * sizeof(Tensor*));
    int idx = 0;

    _build_topo(root, &sorted, &idx, &max_nodes);
    
    if(root->grad){
        int user_set_grad = 0;

        for(int i=0; i<root->size; i++){
            if(root->grad[i]){
                user_set_grad = 1;
                break;
            }
        }

        if(!user_set_grad) for(int i=0; i<root->size; i++) root->grad[i] = 1.0;
    }

    for(int i = idx - 1; i>=0; i--){
        Tensor* t = sorted[i];
        if(t->op_backward) t->op_backward(t);
    }

    for(int i=0; i<idx; i++) sorted[i]->visited = 0;

    free(sorted);
}

void print_tensor(Tensor* v){

    printf("[\n");
    for(int i = 0; i<v->size; i++){
        if(i != v->size-2) printf("[%f]\n",v->data[i]);
        else printf("[%f]",v->data[i]);
    }
    printf("]\n");
}

void shape(Tensor* v){
    printf("(%d, 1)\n", v->size);
}