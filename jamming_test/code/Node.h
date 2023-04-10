#ifndef _NODE_H
#define _NODE_H

class CSuperball;

class CNode {
 public:
  CNode(CSuperball* pS, double* pbc);
  ~CNode(void);

 public:
  CNode* prev;
  CNode* next;
  CSuperball* ps;
  double* PBC;
};

#endif
