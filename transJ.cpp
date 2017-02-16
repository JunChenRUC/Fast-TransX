//============================================================================
// Name        : TransE.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <map>
#include <set>
#include <vector>

using namespace std;

const float pi = 3.141593;
const float E = 2.718283;

int transeThreads = 4;
int transeTrainTimes = 10000;
int nbatches = 1;
int dimension = 100;
int measure = 2;
float transeAlpha = 0.0005;
float margin_positive = 0.1, margin_negative_low = 0.6, margin_negative_high = 0.5;
float alpha = 0.01;

string inputPath = "D://kb_data/instance_processed/";
string outputPath = "D://kb_data/instance_processed/TransE/";

struct Triple {
	int h, r, t;
};

Triple *trainList;

/*
	There are some math functions for the program initialization.
*/

unsigned long long *next_random;

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

float rand(float min, float max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do{
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

void norm(float * con) {
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if(x > 1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

float sqr(float x)
{
    return x*x;
}

/*
	Read triples from the training file.
*/

int relationTotal = 0, entityTotal = 0, tripleTotal = 0;

float *relationVec, *entityVec;
map<int,string> id2entity,id2relation;
map<string,int> relation2id,entity2id;
map<int,int> entity2type;
//float *relationVecDao, *entityVecDao;
map<int,map<int,set<int> > > entity2relation2entity_out, entity2relation2entity_in;
map<int,set<int> > relation2entity_left_tmp, relation2entity_right_tmp;
map<int,vector<int> > relation2entity_left, relation2entity_right;


int train_size_negative_low, train_size_negative_high, train_size_postive;
float distance_postive, distance_negative_low, distance_negative_high;

void init() {
	int x,y;
	char buf1[1000], buf2[1000], buf3[1000];

	//initial entity 2 id
    FILE* f1 = fopen((inputPath + "entity2id.txt").c_str(),"r");
	while (fscanf(f1,"%s%d%d", buf1, &x, &y) == 3){
		string name = buf1;
		entity2id[name] = x;
		id2entity[x] = name;
		entity2type[x] = y;
		entityTotal ++;
	}
	fclose(f1);
	
	//initial entity 2 vector
	entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec + i*dimension);
	}
	printf("Reading file: %s, fishing initialization, entity number is %d\n", (inputPath + "entity2id.txt").c_str(), entityTotal);
	
	//initial relation 2 id
	FILE* f2 = fopen((inputPath + "relation2id.txt").c_str(),"r");
	while (fscanf(f2,"%s%d", buf1, &x) == 2){
		string name = buf1;
		relation2id[name] = x;
		id2relation[x] = name;
		relationTotal ++;
	}
	fclose(f2);
	
	//initial relation 2 vector
	relationVec = (float *)calloc(relationTotal * dimension, sizeof(float));
	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
			relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	}
	printf("Reading file: %s, fishing initialization, relation number is %d\n", (inputPath + "relation2id.txt").c_str(), relationTotal);
	
	//initial triple
    FILE* f3 = fopen((inputPath + "train.txt").c_str(),"r");
    while (fscanf(f3,"%s%s%s", buf1, buf2, buf3) == 3){
		tripleTotal ++;
	}
    fclose(f3);
	
	int count = 0;
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	
	FILE* f4 = fopen((inputPath + "train.txt").c_str(),"r");
	while (fscanf(f4,"%s%s%s", buf1, buf2, buf3) == 3){
		string head = buf1, tail = buf2, relation = buf3;
		int headId = entity2id[head];
		int tailId = entity2id[tail];
		int relationId = relation2id[relation];
		trainList[count].h = headId;
		trainList[count].t = tailId;
		trainList[count].r = relationId;
		count ++;
		
		entity2relation2entity_out[headId][relationId].insert(tailId);
		entity2relation2entity_in[tailId][relationId].insert(headId);
		
		relation2entity_left_tmp[relationId].insert(headId);
		relation2entity_right_tmp[relationId].insert(tailId);
	}
	fclose(f4);
	printf("Reading file: %s, triple number is %d\n", (inputPath + "train.txt").c_str(), tripleTotal);
	
	for(int i = 0; i < relationTotal; i ++){
		vector<int> vector_left(relation2entity_left_tmp[i].begin(), relation2entity_left_tmp[i].end());
		relation2entity_left[i] = vector_left;
		
		vector<int> vector_right(relation2entity_right_tmp[i].begin(), relation2entity_right_tmp[i].end());
		relation2entity_right[i] = vector_right;
	}
	
	printf("Statistics: \n");
	for(int i = 0; i < relationTotal; i ++){
		printf("relation Id=%d\tleft size=%d\tright size=%d: \n", i, relation2entity_left[i].size(), relation2entity_right[i].size());
	}
}

/*
	Training process of transE.
*/

int transeLen;
int transeBatch;
float res;

float calc_sum(int h, int t, int r) {
	int last_h = h * dimension, last_t = t * dimension, last_r = r * dimension;
	
	float sum = 0;
	for (int i = 0; i < dimension; i ++) {
		if(measure == 1)
			sum += fabs(entityVec[last_t + i] - entityVec[last_h + i] - relationVec[last_r + i]);
		else if(measure == 2)
			sum += sqr(entityVec[last_t + i] - entityVec[last_h + i] - relationVec[last_r + i]);
	}
	if(measure == 1)
		;
	else if(measure == 2)
		sum = sqrt(sum);
	
	return sum;
}

void make_close(int h, int t, int r) {
	int last_h = h * dimension, last_t = t * dimension, last_r = r * dimension;
	for (int i =0; i  < dimension; i ++) {
		float x = (entityVec[last_t + i] - entityVec[last_h + i] - relationVec[last_r + i]);
		
		if (x > 0)
			x = -transeAlpha;
		else
			x = transeAlpha;
		
		relationVec[last_r + i] -= x;
		entityVec[last_h + i] -= x;
		entityVec[last_t + i] += x;
	}
	norm(entityVec + last_h);
	norm(entityVec + last_t);
	norm(relationVec + last_r);
}

void make_away(int h, int t, int r) {
	int last_h = h * dimension, last_t = t * dimension, last_r = r * dimension;
	for (int i =0; i  < dimension; i ++) {
		float x = (entityVec[last_t + i] - entityVec[last_h + i] - relationVec[last_r + i]);
		
		if (x > 0)
			x = transeAlpha;
		else
			x = -transeAlpha;
		
		relationVec[last_r + i] -=  x;
		entityVec[last_h + i] -= x;
		entityVec[last_t + i] += x;
	}	
	norm(entityVec + last_h);
	norm(entityVec + last_t);
	norm(relationVec + last_r);
}

void train_kb_positive(int id, int h, int t, int r, float threshold) {
	float distance = calc_sum(h, t, r);	
	float margin = margin_positive - margin_positive * threshold;
	
	if (distance > margin) {
		distance_postive += distance;
		train_size_postive ++;
		//res += distance - margin;
		make_close(h, t, r);
	}
}

int negative_head_low(int id, int t, int r) {
	int h = 0;
	int size = relation2entity_left[r].size();
	do{
		h = relation2entity_left[r].at(rand_max(id, size));
	} while(entity2relation2entity_in[t][r].count(h) > 0);
	
	return h;
}

int negative_head_high(int id, int t, int r) {
	int r_negative;
	do{
		r_negative = rand_max(id, relationTotal);
	} while(r_negative == r);
	int size = relation2entity_left[r_negative].size();
	int h = 0;
	do{
		h = relation2entity_left[r_negative].at(rand_max(id, size));
	} while(entity2relation2entity_in[t][r_negative].count(h) > 0);
	
	return h;
}

int negative_tail_low(int id, int h, int r) {
	int t = 0;
	int size = relation2entity_right[r].size();
	do{
		t = relation2entity_right[r].at(rand_max(id, size));
	} while(entity2relation2entity_out[h][r].count(t) > 0);
	
	return t;
}

int negative_tail_high(int id, int h, int r) {
	int r_negative;
	do{
		r_negative = rand_max(id, relationTotal);
	} while(r_negative == r);
	int size = relation2entity_right[r_negative].size();
	int t = 0;
	do{
		t = relation2entity_right[r_negative].at(rand_max(id, size));
	} while(entity2relation2entity_out[h][r_negative].count(t) > 0);
	
	return t;
}

void train_kb_negative(int id, int h, int t, int r, float threshold, int flag) {
	float distance, margin;
	//corrupt tail
	if(flag == 0)
		t = negative_tail_low(id, h, r);	
	else if(flag == 1)
		h = negative_head_low(id, t, r);
	
	distance = calc_sum(h, t, r);
	//margin = margin_positive + (margin_negative_low - margin_positive) * threshold;
	margin = 2 * (margin_positive - margin_positive * threshold);
	
	if (distance < margin) {
		distance_negative_low += distance;
		train_size_negative_low ++;
		//res += margin - distance;
		make_away(h, t, r);
	}
	
	//corrupt tail
	if(flag == 0)
		t = negative_tail_high(id, h, r);
	else if(flag == 1)
		h = negative_head_high(id, t, r);
	
	distance = calc_sum(h, t, r);
	margin = margin_negative_high;
	
	if (distance < margin) {
		distance_negative_high += distance;
		train_size_negative_high ++;
		//res += margin - distance;
		make_away(h, t, r);
	}
}

float threshold_min, threshold_max;
int negative_size_head, negative_size_tail;

void* transetrainMode(void *con) {
	int id = (unsigned long long)(con);
	next_random[id] = rand();
	
	int m = transeBatch / transeThreads;
	int n = transeBatch % transeThreads;
	for (int i = m * id; i < m * (id + 1) + ((id + 1) == transeThreads ? n : 0); i ++) {
		int h = trainList[i].h, t = trainList[i].t, r = trainList[i].r;
		
		int pr = 500;
		if(randd(id) % 1000 < pr /*&& !(r == 2 || r == 5)*/) {//generate negative tail
			//negative_size_tail ++;
			//if(entity2relation2entity_out[h][r].size() == 0)
				//printf("error");
			//float threshold = 1.0 / log(entity2relation2entity_out[h][r].size() + E - 1);// / log(relation2entity_right[r].size());
			float threshold = 1.0 / log(entity2relation2entity_out[h][r].size() + E - 1 + alpha);
			
			if(threshold > threshold_max)
				threshold_max = threshold;
			if(threshold < threshold_min)
				threshold_min = threshold;
			
			train_kb_positive(id, h, t, r, threshold);	
			train_kb_negative(id, h, t, r, threshold, 0);
			//
		}else{//generate negative head
			//negative_size_head ++;
			//if(entity2relation2entity_in[t][r].size() == 0)
				//printf("error");
			//float threshold = 1.0 / log(entity2relation2entity_in[t][r].size() + E - 1); // / log(relation2entity_left[r].size());
			float threshold = 1.0 / log(entity2relation2entity_in[t][r].size() + E - 1 + alpha);
			
			if(threshold > threshold_max)
				threshold_max = threshold;
			if(threshold < threshold_min)
				threshold_min = threshold;
			
			train_kb_positive(id, h, t, r, threshold);	
			train_kb_negative(id, h, t, r, threshold, 1);
		}			
	}
}

void* train_transe(void *con) {
	transeLen = tripleTotal;
	transeBatch = transeLen / nbatches;
	next_random = (unsigned long long *)calloc(transeThreads, sizeof(unsigned long long));
	for (int epoch = 0; epoch < transeTrainTimes; epoch++) {
		//res = 0;
		train_size_postive = 0, train_size_negative_low = 0, train_size_negative_high = 0;
		distance_postive = 0, distance_negative_low = 0, distance_negative_high = 0;
		negative_size_head =0, negative_size_tail = 0; 
		threshold_min = 1, threshold_max = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(transeThreads * sizeof(pthread_t));
			for (int a = 0; a < transeThreads; a++)
				pthread_create(&pt[a], NULL, transetrainMode,  (void*)a);
			for (int a = 0; a < transeThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
		printf("epoch %d\tpostive size=%d\tpositve distance=%f\tnegative size low=%d\tnegative distance low=%f\tnegative size high=%d\tnegative distance high=%f\n", epoch, train_size_postive, distance_postive/train_size_postive, train_size_negative_low, distance_negative_low/train_size_negative_low, train_size_negative_high, distance_negative_high/train_size_negative_high);
		//printf("negative head size=%d\t negative tail size=%d\n", negative_size_head, negative_size_tail);
	}
}

/*
	Get the results of transE.
*/

void out_transe() {
	FILE* f1 = fopen((outputPath + "relation2vec.bern").c_str(), "w");
	for (int i = 0; i < relationTotal; i ++) {
		int last = dimension * i;
		for (int j = 0; j < dimension; j ++)
			fprintf(f1, "%.6f\t", relationVec[last + j]);
		fprintf(f1,"\n");
	}
	fclose(f1);
	
	FILE* f2 = fopen((outputPath + "entity2vec.bern").c_str(), "w");
	for (int  i = 0; i < entityTotal; i ++) {
		int last = i * dimension;
		for (int j = 0; j < dimension; j ++)
			fprintf(f2, "%.6f\t", entityVec[last + j] );
		fprintf(f2,"\n");
	}
	fclose(f2);
}

/*
	Main function
*/

int main() {
	init();
	train_transe(NULL);
	out_transe();
	return 0;
}
