#include<stdlib.h>
#include<stdio.h>

#include "io.h"
#include "stensor.h"
#include "graph.h"


extern void _build_topo(Tensor* root, Tensor*** sorted, int* idx, int* cap);

void _reset_visited_local(Tensor* t){
    if(!t || t->visited) return;

    t->visited = 1;
    _reset_visited_local(t->left);
    _reset_visited_local(t->right);
}

int get_id(Tensor** list, int count, Tensor* target){
    if(!target) return -1;

    for(int i=0; i<count; i++){
        if(list[i] == target) return i;
    }

    return -1;
}


void save_model(Tensor* root, int max_nodes,const char* filename){

    Tensor** sorted = (Tensor**)malloc(max_nodes * sizeof(Tensor*));
    int idx = 0;

    _reset_visited_local(root);
    
    _build_topo(root, &sorted, &idx, &max_nodes);
    
    _reset_visited_local(root);

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("[IO] Error: Could not open file %s\n", filename);
        free(sorted);
        return;
    }

    fprintf(fp, "[\n");

    for(int i=0; i<max_nodes; i++){
        Tensor* t = sorted[i];

        fprintf(fp, "{\n");
        fprintf(fp, "   \"id\": %d,\n", i);
        fprintf(fp, "   \"type\": %d,\n", t->type);
        fprintf(fp, "   \"size\": %d, \n", t->size);

        int left_id = get_id(sorted, max_nodes,  t->left);
        int right_id = get_id(sorted, max_nodes,  t->right);
    
        fprintf(fp, "   \" left_id\": %d,\n", left_id);
        fprintf(fp, "   \" right_id\": %d,\n", right_id);
    
        fprintf(fp, "   \" data\": [");
        if(t->data){
            for(int k=0; k < t->size; k++){
                fprintf(fp, "%.17g%s", t->data[k], (k < t->size - 1) ? ", ": "");
            }
        }
        fprintf(fp, "],\n");

        
    }
}