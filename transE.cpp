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

const float pi = 3.141592653589793238462643383;

int transeThreads = 2;
int transeTrainTimes = 3000;
int nbatches = 1;
int dimension = 50;
float transeAlpha = 0.001;
float margin = 1;

string inputPath = "D://kb_data/instance_processed/";
string outputPath = "D://kb_data/instance_processed/TransE/";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
};

Triple *trainHead, *trainTail, *trainList;

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

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
	do {
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
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

int relationTotal, entityTotal, tripleTotal;
float *relationVec, *entityVec;
map<int,string> id2entity,id2relation;
map<string,int> relation2id,entity2id;
map<int,int> entity2type;

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
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
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
		trainHead[count].h = headId;
		trainHead[count].t = tailId;
		trainHead[count].r = relationId;
		trainTail[count].h = headId;
		trainTail[count].t = tailId;
		trainTail[count].r = relationId;
		
		count ++;
	}
	fclose(f4);
	printf("Reading file: %s, triple number is %d\n", (inputPath + "train.txt").c_str(), tripleTotal);
	
	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());

	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(rigHead));
	memset(rigTail, -1, sizeof(rigTail));
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;
}

/*
	Training process of transE.
*/

int transeLen;
int transeBatch;
float res;

float calc_sum(int e1, int e2, int rel) {
	float sum=0;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimension;
	for (int ii=0; ii < dimension; ii++) {
		sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);	
	}
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimension;
	int lastb1 = e1_b * dimension;
	int lastb2 = e2_b * dimension;
	int lastbr = rel_b * dimension;
	for (int ii=0; ii  < dimension; ii++) {
		float x;
		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
		if (x > 0)
			x = -transeAlpha;
		else
			x = transeAlpha;
		relationVec[lastar + ii] -= x;
		entityVec[lasta1 + ii] -= x;
		entityVec[lasta2 + ii] += x;
		x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
		if (x > 0)
			x = transeAlpha;
		else
			x = -transeAlpha;
		relationVec[lastbr + ii] -=  x;
		entityVec[lastb1 + ii] -= x;
		entityVec[lastb2 + ii] += x;
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	float sum1 = calc_sum(e1_a, e2_a, rel_a);
	float sum2 = calc_sum(e1_b, e2_b, rel_b);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	}
}

int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* transetrainMode(void *con) {
	int id;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (int k = transeBatch / transeThreads; k >= 0; k--) {
		int j;
		int i = rand_max(id, transeLen);
		int pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r);
		}
		norm(relationVec + dimension * trainList[i].r);
		norm(entityVec + dimension * trainList[i].h);
		norm(entityVec + dimension * trainList[i].t);
		norm(entityVec + dimension * j);
	}
}

void* train_transe(void *con) {
	transeLen = tripleTotal;
	transeBatch = transeLen / nbatches;
	next_random = (unsigned long long *)calloc(transeThreads, sizeof(unsigned long long));
	for (int epoch = 0; epoch < transeTrainTimes; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(transeThreads * sizeof(pthread_t));
			for (int a = 0; a < transeThreads; a++)
				pthread_create(&pt[a], NULL, transetrainMode,  (void*)a);
			for (int a = 0; a < transeThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
		printf("epoch %d %f\n", epoch, res);
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
