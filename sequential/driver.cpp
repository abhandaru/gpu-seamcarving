//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#include "driver.h"

using std::cout;
using std::endl;


int main(int argc, char const *argv[]) {

  Image image("../images/boat-16.bmp");

  // Simple test to see if the image was loaded correctly.
  cout << ">> pixel (0, 1): " << endl;
  const RGBQuad& p = image[0][1];
  cout << "   R: " << (int)p.red << endl;
  cout << "   G: " << (int)p.green << endl;
  cout << "   B: " << (int)p.blue << endl;

  Seamcarver seamcarver(&image);
  seamcarver.removeSeam();

  // Clean up and return normally.
  return 0;
}
