package tests.npeExists;

public class StaticField {
	public StaticField f;
	public static StaticField s;
	public static void main(String[] args) {
		StaticField h = new StaticField();
		StaticField.s = h;
		h = StaticField.s.f;
		h = h.f; // NPE
	}
}
