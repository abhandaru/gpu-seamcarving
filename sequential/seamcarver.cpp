//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#include "seamcarver.h"

using std::cout;
using std::endl;


Seamcarver::Seamcarver(Image* image) {
  cout << ">> init seamcarver" << endl;
  _image = image;
}


Seamcarver::~Seamcarver() {

}


// Removes 1 seam.
void Seamcarver::removeSeam() {
  Energies energies(_image);

}
