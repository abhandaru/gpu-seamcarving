//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#include "image.h"


// namespace shortcuts
using std::cout;
using std::endl;


Image::Image(const char* path) {
  cout << ">> loading: " << path << endl;
  cout << "   file header size: " << sizeof(BitmapFileHeader) << endl;
  cout << "   info header size: " << sizeof(BitmapInfoHeader) << endl;

  FILE* file = fopen(path, "rb");
  if (file) {
    readBitmap(file);
    fclose(file);
  } else {
    cout << ">> unable to open: " << path << endl;
  }
}


Image::~Image() {
  delete _pixels;
}


size_t Image::width() const {
  return _width;
}


size_t Image::height() const {
  return _height;
}


const RGBQuad& Image::get(size_t row, size_t col) const {
  size_t index = row * _width + col;
  return _pixels[index];
}


//
// Private methods
//

void Image::readBitmap(FILE* file) {
  fread(&_fileHeader, sizeof(BitmapFileHeader), 1, file);
  fread(&_infoHeader, sizeof(BitmapInfoHeader), 1, file);
  _width = _infoHeader.biWidth;
  _height = -_infoHeader.biHeight; // Why do we have to negate?

  // Read in the pixel data.
  size_t size = _width * _height;
  _pixels = new RGBQuad[size];
  fread(_pixels, sizeof(RGBQuad), size, file);

  // Print out some info.
  cout << ">> parsing bitmap ..." << endl;
  cout << "   width: " << _width<< endl;
  cout << "   height: " << _height << endl;
}
