package tests.npeExists;
public class Loop {
  public Loop y;

  public static void main(String[] args) {
    Loop h = new Loop();
    int x = 1;
    while (x < 10){
      h = h.y;
    }
    h.foo();
  }

  public void foo(){
    int x = 4;
    x = x + x;
  }
}
