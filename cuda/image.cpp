//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#include "image.h"


// namespace shortcuts
using std::cout;
using std::endl;
using std::vector;


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


int Image::width() const {
  return _width;
}


int Image::height() const {
  return _height;
}


const RGBQuad& Image::get(int row, int col) const {
  int index = row * _width + col;
  return _pixels[index];
}


const RGBQuad* Image::operator [](int i) const {
  return _pixels + (i * _width);
};


void Image::removeSeam(vector<int>& seam) {
  int length = _width * _height;
  int num_removed = 0;
  for (int i = 0; i < length; i++) {
    int row = num_removed;
    int col = seam[row];

    int index = row * _width + col;
    if (i == index) {
      num_removed++;
    }

    _pixels[i] = _pixels[i + num_removed];
  }

  // Update width.
  _width--;
  _info_header.biWidth = _width;
}


// Directions on how to do this are here:
// http://stackoverflow.com/questions/18838553/c-how-to-create-a-bitmap-file
void Image::save(const char* path) const {
  FILE* file = fopen(path, "wb");
  fwrite(&_file_header, sizeof(BitmapFileHeader), 1, file);
  fwrite(&_info_header, sizeof(BitmapInfoHeader), 1, file);
  fwrite(_pixels, sizeof(RGBQuad), _width * _height, file);
  fclose(file);
}


//
// Private methods
//

void Image::readBitmap(FILE* file) {
  fread(&_file_header, sizeof(BitmapFileHeader), 1, file);
  fread(&_info_header, sizeof(BitmapInfoHeader), 1, file);
  _width = _info_header.biWidth;
  _height = -_info_header.biHeight; // Why do we have to negate?

  // Read in the pixel data.
  int size = _width * _height;
  _pixels = new RGBQuad[size];
  fread(_pixels, sizeof(RGBQuad), size, file);

  // Print out some info.
  cout << ">> parsing bitmap ..." << endl;
  cout << "   width: " << _width << endl;
  cout << "   height: " << _height << endl;
}
