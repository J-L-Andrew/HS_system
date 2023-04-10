#include "Node.h"

#include <stdlib.h>

CNode::CNode(CSuperball* pS, double* pbc) {
  prev = NULL;
  next = NULL;
  ps = pS;
  PBC = pbc;
}
CNode::~CNode(void) {
  if (PBC != NULL) delete[] PBC;
}
