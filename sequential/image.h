//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <iostream>
#include <stdio.h>

#include "bitmap.h"


class Image {
 public:
  Image(const char* path);
  ~Image();
  void save(const char* path);

  // getters
  size_t width() const;
  size_t height() const;
  const RGBQuad& get(size_t row, size_t col) const;

 private:
  void readBitmap(FILE* file);

  // data
  size_t _width;
  size_t _height;
  RGBQuad* _pixels;

  // TODO: Should use some sort of polymorphism here.
  BitmapFileHeader _fileHeader;
  BitmapInfoHeader _infoHeader;
};

#endif