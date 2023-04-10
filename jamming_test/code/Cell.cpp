#include "Cell.h"

#include "stdio.h"

CCell::CCell() {
  for (int i = 0; i < 8; i++) head[i] = NULL;
}

CCell::~CCell() {
  CNode* pn;
  for (int i = 0; i < 8; i++) {
    pn = head[i];
    while (pn != NULL) {
      if (pn->next != NULL) {
        pn = pn->next;
        delete pn->prev;
      } else {
        delete pn;
        pn = NULL;
      }
    }
    head[i] = pn;
  }
}
void CCell::Add(CNode* pn, int i) {
  if (head[i] == NULL)
    head[i] = pn;
  else {
    pn->next = head[i];
    head[i]->prev = pn;
    head[i] = pn;
  }
}