package tests.npeExists;
public class Switch {
  public Switch y;

  public static void main(String[] args) {
    Switch h = new Switch();
    int x = 3;
    switch (x){
    	case 1: {h = h.y; break;}
    	case 2: {h = new Switch(); break;}
    	case 3: {h.y = new Switch(); h = h.y; break;}
    	default: {x = x+1;}
    }
    h.foo();
  }

  public void foo(){
    int x = 4;
  }
}
