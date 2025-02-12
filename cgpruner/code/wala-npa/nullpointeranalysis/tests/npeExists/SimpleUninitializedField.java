package tests.npeExists;
public class SimpleUninitializedField {
  public SimpleUninitializedField y;

  public static void main(String[] args) {
    SimpleUninitializedField h = new SimpleUninitializedField();
    SimpleUninitializedField s = h.y;
    int x = s.foo();
  }

  public int foo(){
    return 3;
  }
}
