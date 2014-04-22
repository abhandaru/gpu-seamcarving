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
  Image image("../images/boat-1024.bmp");

  // Simple test to see if the image was loaded correctly.
  cout << ">> pixel (0, 1): " << endl;
  const RGBQuad& p = image[0][1];
  cout << "   R: " << (int)p.red << endl;
  cout << "   G: " << (int)p.green << endl;
  cout << "   B: " << (int)p.blue << endl;

  // Clock start time.
  int num_seams = atoi(argv[2]);
  cout << ">> init seamcarver" << endl;
  cout << "   removing " << num_seams << " seams ..." << endl;
  clock_t begin = clock();

  // Remove seams.
  Seamcarver seamcarver(&image);
  seamcarver.removeSeams(num_seams);

  // Clock end time.
  clock_t end = clock();
  clock_t exec_time = end - begin;
  cout << ">> Execution time ..." << endl;
  cout << "   cycles: " << exec_time << endl;
  cout << "   time: " << ((double)exec_time /  CLOCKS_PER_SEC) << "s" << endl;

  // Clean up and return normally.
  image.save("../outputs/sequential.bmp");
  return 0;
}
