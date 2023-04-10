// Custom output example code
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include <fstream>
#include <iostream>

#include "../src/voro++.hh"
using namespace std;
using namespace voro;

// Set up constants for the container geometry
double x_min = 0.0, x_max = 8.39822093;
double y_min = 0.0, y_max = 8.39822093;
double z_min = 0.0, z_max = 8.39822093;
int n_x = 8, n_y = 8, n_z = 8;

int main() {
  int i, istart;
  int ntemp, itemp;
  double Lb;
  char input_file[100];

  ifstream input("../../vorofile/boundary.txt");
  
  ntemp = 1;
  for (i = 0; i < ntemp; i++) {
    input >> itemp >> Lb;
    printf("%d\t%d\t%lf\n", i, itemp, Lb);

    x_max = Lb;
    y_max = Lb;
    z_max = Lb;
    container con(x_min, x_max, y_min, y_max, z_min, z_max, n_x, n_y, n_z, true,
                  true, true, 8);
    sprintf(input_file, "../../vorofile/Surfpoint%d.txt", itemp);
    con.import(input_file);
    sprintf(input_file, "../../vorofile/V_voro%d.txt", itemp);
    con.print_custom("%i\t%x\t%y\t%z\t%v", input_file);
  }
  input.close();
  return 0;
}
