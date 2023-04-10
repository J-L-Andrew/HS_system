#ifndef _CELL_H
#define _CELL_H

#include "Node.h"

class CCell {
 public:
  CCell();
  ~CCell();

 public:
  CNode* head[8];

 public:
  void Add(CNode* pn, int i);
};

#endif