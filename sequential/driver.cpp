//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#include "driver.h"

using std::cout;
using std::endl;


int main(int argc, char const *argv[]) {

  Image image("../images/boat-128.bmp");
  const RGBQuad& p = image.get(0, 0);
  cout << "R: " << (int)p.red << endl;
  cout << "G: " << (int)p.green << endl;
  cout << "B: " << (int)p.blue << endl;

  Seamcarver seamcarver;

  return 0;
}
