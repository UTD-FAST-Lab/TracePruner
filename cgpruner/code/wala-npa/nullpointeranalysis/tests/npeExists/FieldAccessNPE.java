package tests.npeExists;
public class FieldAccessNPE {
  public FieldAccessNPE y;

  public static void main(String[] args) {
    FieldAccessNPE h = new FieldAccessNPE();
    FieldAccessNPE s = h.y;
    s.y = h;
    FieldAccessNPE t = s.y;
  }

  public int foo(){
    return 3;
  }
}
