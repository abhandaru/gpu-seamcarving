//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#include "energies.h"

Energies::Energies(Image* image) {
  _width = image->width();
  _height = image->height();
  _image = image;
  _energies = new float[_width * _height];
}


Energies::~Energies() {
  delete _energies;
}


//
// Filtering options
//
void Energies::gradient() {
  for (int row = 0; row < _height; row++) {
    for (int col = 0; col < _width; col++) {
      const RGBQuad& current = _image->get(row, col);
      const RGBQuad& right = _image->get(row, col + 1);
      const RGBQuad& down = _image->get(row + 1, col);

      // Compute gradient.
      // This is a vector operation, so there is room for improvement.
      float dxRed = right.red - current.red;
      float dxGreen = right.green - current.green;
      float dxBlue = right.blue - current.blue;
      float dyRed = down.red - current.red;
      float dyGreen = down.green - current.green;
      float dyBlue = down.blue - current.blue;

      // Sum the squares of differences.
      float dx2 = dxRed*dxRed + dxGreen*dxGreen + dxBlue*dxBlue;
      float dy2 = dyRed*dxRed + dyGreen*dxGreen + dyBlue*dxBlue;
      float grad = (float) sqrt(dx2 + dy2);

      // Store results.
      int index = row * _width + col;
      _energies[index] = grad;
    }
  }
}


//
// Getters and operators
//
size_t Energies::width() const {
  return _width;
}


size_t Energies::height() const {
  return _height;
}


float Energies::get(int row, int col) const {
  if (row < 0 || row >= _height ||
      col < 0 || col >= _width) {
    return 0.0;
  }

  // Common case.
  size_t index = row * _width + col;
  return _energies[index];
}


const float* Energies::operator [](int i) const {
  return _energies + (i * _width);
};
