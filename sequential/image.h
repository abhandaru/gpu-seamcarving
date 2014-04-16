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
  int width() const;
  int height() const;
  const RGBQuad& get(int row, int col) const;
  const RGBQuad* operator [](int i) const; // for convenient access

 private:
  void readBitmap(FILE* file);

  // data
  int _width;
  int _height;
  RGBQuad* _pixels;

  // TODO: Should use some sort of polymorphism here.
  BitmapFileHeader _fileHeader;
  BitmapInfoHeader _infoHeader;
};

#endif