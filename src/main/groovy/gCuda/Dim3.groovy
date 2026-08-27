package gCuda

class Dim3 {
  int x,y,z

  Dim3() {
    this.x = 1
    this.y = 1
    this.z = 1
  }

  @Override
  String toString() {
    return "[$x, $y, $z]"
  }
}
