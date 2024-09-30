package nullpointeranalysis;

import java.util.ArrayList;

public class Constraint {
	/* CONSTRAINTS LIST */
	public static ArrayList<SimpleConstraint<Val>> 
		valSimpleConstraints;
	public static ArrayList<SimpleConstraint<Def>> 
		defSimpleConstraints;
	public static ArrayList<ConditionalAssignmentConstraint> 
		conditionalAssgnConstraints;
	public static ArrayList<ConditionalAssignmentConstraintType2> 
		type2conditionalAssgnConstraints;
	public static ArrayList<ConditionalConstraint> 
		conditionalConstraints;
	public static ArrayList<AntiConditionalConstraint> 
		antiConditionalConstraints;
	
	static {
		valSimpleConstraints = new ArrayList<SimpleConstraint<Val>>();
		defSimpleConstraints = new ArrayList<SimpleConstraint<Def>>();
		conditionalAssgnConstraints = new ArrayList<ConditionalAssignmentConstraint>();
		type2conditionalAssgnConstraints = new ArrayList<ConditionalAssignmentConstraintType2>();
		conditionalConstraints = new ArrayList<ConditionalConstraint>();
		antiConditionalConstraints = new ArrayList<AntiConditionalConstraint>();
	}
}

/* CONSTRAINTS CLASS DEFINITIONS */

// simple constraint of the form "greaterElement >= lesserElement"
class SimpleConstraint<T> extends Constraint{
	T lesserElement;
	T greaterElement;
	public SimpleConstraint (T a, T b){
		lesserElement = a;
		greaterElement = b;
	}
}

//Constraint of the form "if checkedValue == expectedValue then "lesserElement <= greaterElement"
class ConditionalConstraint extends Constraint{
	Def checkedValue;
	Def expectedValue;
	Def lesserElement;
	Def greaterElement;
	public ConditionalConstraint(Def a, Def b, Def c, Def d) {
		checkedValue = a;
		expectedValue = b;
		lesserElement = c;
		greaterElement = d;
	}
}

//Constraint of the form "if checkedValue == expectedValue then lhs=rhs"
class ConditionalAssignmentConstraint extends Constraint{
	Def checkedValue;
	Def expectedValue;
	Val lhs;
	Val rhs;
	public ConditionalAssignmentConstraint(Def a, Def b, Val c, Val d) {
		checkedValue = a;
		expectedValue = b;
		lhs = c;
		rhs = d;
	}
	
}

//Constraint of the form "if checkLesser < checkGreater then lhs=rhs"
class ConditionalAssignmentConstraintType2 extends Constraint{
	Val checkLesser;
	Val checkGreater;
	Def lhs;
	Def rhs;
	public ConditionalAssignmentConstraintType2(Val a, Val b, Def c, Def d) {
		checkLesser = a;
		checkGreater = b;
		lhs = c;
		rhs = d;
	}
	
}

// constraint of the form "if elementToCheck!=checkAgainst, greaterElement >= lesserElement"
class AntiConditionalConstraint extends Constraint{
	Val elementToCheck;
	Val checkAgainst;
	Val lesserElement;
	Val greaterElement;
	public AntiConditionalConstraint (Val a, Val b, Val c, Val d){
		elementToCheck = a;
		checkAgainst = b;
		lesserElement = c;
		greaterElement = d;
	}
}