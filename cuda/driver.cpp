//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#include "driver.h"

using std::cout;
using std::endl;


int main(int argc, char const *argv[]) {
  // Parse args
  if (argc < 3) {
    cout << "Usage: driver.out -n $num_seams" << endl;
    exit(1);
  }

  // Load a bitmap image.
  Image image("../images/boat-512.bmp");

  // Simple test to see if the image was loaded correctly.
  cout << ">> pixel (0, 1): " << endl;
  const RGBQuad& p = image[0][1];
  cout << "   R: " << (int)p.red << endl;
  cout << "   G: " << (int)p.green << endl;
  cout << "   B: " << (int)p.blue << endl;

  // Remove seams.
  int num_seams = atoi(argv[2]);
  Seamcarver seamcarver(&image);
  seamcarver.removeSeams(num_seams);

  // Clean up and return normally.
  image.save("../outputs/test.bmp");
  return 0;
}
