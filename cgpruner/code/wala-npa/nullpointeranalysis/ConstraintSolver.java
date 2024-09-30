package nullpointeranalysis;

public class ConstraintSolver {
	public static void solve() {
		boolean somethingChanged = true;
		// Keep propagating the values in the constraints
		// until nothing changes (i.e. fixed point)
		int num_iter = 0;
		while(somethingChanged && num_iter < Constants.maxIterations) {
			num_iter += 1;
			somethingChanged = false;
			for (SimpleConstraint<Val> sc : Constraint.valSimpleConstraints) {
				if (sc.lesserElement.compareTo(sc.greaterElement) > 0) {
					somethingChanged = true;
					setToMeetValue(sc.greaterElement, sc.lesserElement);
				}
			}
			
			for (SimpleConstraint<Def> sc : Constraint.defSimpleConstraints) {
				if (sc.lesserElement.compareTo(sc.greaterElement) > 0) {
					somethingChanged = true;
					sc.greaterElement.base = sc.lesserElement.base;
				}
			}
			
			for (ConditionalAssignmentConstraint cac : Constraint.conditionalAssgnConstraints) {
				if (cac.checkedValue.equals(cac.expectedValue)) {
					if (!cac.lhs.equals(cac.rhs)) {
						somethingChanged = true;
						cac.lhs.baseRepresentation = cac.rhs.baseRepresentation;
						cac.lhs.rawClass = cac.rhs.rawClass;
					}
				}
			}
			
			for (ConditionalAssignmentConstraintType2 cact2 : Constraint.type2conditionalAssgnConstraints) {
				if (cact2.checkLesser.compareTo(cact2.checkGreater) <= 0) {
					if (!cact2.lhs.equals(cact2.rhs)) {
						somethingChanged = true;
						cact2.lhs.base = cact2.rhs.base;
					}
				}
			}
 			
			for (ConditionalConstraint cc : Constraint.conditionalConstraints) {
				if (cc.checkedValue.equals(cc.expectedValue)) {
					if (cc.lesserElement.compareTo(cc.greaterElement) > 0) {
						somethingChanged = true;
						cc.greaterElement.base = cc.lesserElement.base;
					}
				}
			}
			
			for (AntiConditionalConstraint acc : Constraint.antiConditionalConstraints) {
				if (!acc.elementToCheck.equals(acc.checkAgainst)) {
					if (acc.lesserElement.compareTo(acc.greaterElement) > 0) {
						somethingChanged = true;
						setToMeetValue(acc.greaterElement, acc.lesserElement);
					}
				}
			}
		}
		
		if (Constants.printConstraints) {
			printConstraints();
		}
	}
	
	private static void printConstraints() {
		System.out.println(">>>>>CONSTRAINTS LIST \n");
		System.out.println("SimpleConstraint<Val>");
		for (SimpleConstraint<Val> sc : Constraint.valSimpleConstraints) {
			System.out.println("LesserElement = " + sc.lesserElement.element);
			System.out.println("Val = " + sc.lesserElement.baseRepresentation);
			System.out.println("GreaterElement = " + sc.greaterElement.element);
			System.out.println("Val = " + sc.greaterElement.baseRepresentation);
			System.out.println("");
		}
		System.out.println("-------------");
	
		System.out.println("SimpleConstraint<Def>");
		for (SimpleConstraint<Def> sc : Constraint.defSimpleConstraints) {
			System.out.println("LesserElement = " + sc.lesserElement.element);
			System.out.println("Def = " + sc.lesserElement.base);
			System.out.println("GreaterElement = " + sc.greaterElement.element);
			System.out.println("Def = " + sc.greaterElement.base);
			System.out.println("");
			
		}

		System.out.println("-------------");
		System.out.println("ConditionalAssignmentConstraint");
		for (ConditionalAssignmentConstraint cac : Constraint.conditionalAssgnConstraints) {
			System.out.println("CheckedValue = " + cac.checkedValue.element);
			System.out.println("Def = " + cac.checkedValue.base);
			System.out.println("ExpectedValue = " + cac.expectedValue.element);
			System.out.println("Def = " + cac.expectedValue.base);
			System.out.println("LHS = " + cac.lhs.element);
			System.out.println("Val = " + cac.lhs.baseRepresentation);
			System.out.println("RHS = " + cac.rhs.element);
			System.out.println("Val = " + cac.rhs.baseRepresentation);
			System.out.println("");
			
		}
		
		System.out.println("-------------");
		System.out.println("ConditionalAssignmentConstraintType2");
		for (ConditionalAssignmentConstraintType2 cact2 : Constraint.type2conditionalAssgnConstraints) {
			System.out.println("CheckLesser = " + cact2.checkLesser.element);
			System.out.println("Val = " + cact2.checkLesser.baseRepresentation);
			System.out.println("CheckGreater = " + cact2.checkGreater.element);
			System.out.println("Val = " + cact2.checkGreater.baseRepresentation);
			System.out.println("LHS = " + cact2.lhs.element);
			System.out.println("Def = " + cact2.lhs.base);
			System.out.println("RHS = " + cact2.rhs.element);
			System.out.println("Def = " + cact2.rhs.base);
			System.out.println("");
		}
		
		System.out.println("-------------");
		System.out.println("ConditionalConstraint");
		for (ConditionalConstraint cc : Constraint.conditionalConstraints) {
			System.out.println("CheckedValue = " + cc.checkedValue.element);
			System.out.println("Def = " + cc.checkedValue.base);
			System.out.println("ExpectedValue = " + cc.expectedValue.element);
			System.out.println("Def = " + cc.expectedValue.base);
			System.out.println("LesserElement = " + cc.lesserElement.element);
			System.out.println("Def = " + cc.lesserElement.base);
			System.out.println("GreaterElement = " + cc.greaterElement.element);
			System.out.println("Def = " + cc.greaterElement.base);
			System.out.println("");
		}
		
	}

	// Sets the greater element to be the meet of its old value and the lesser element.
	// Assumes that 'greater'<'lesser'. Else we wouldn't be executing this operation.
	private static void setToMeetValue(Val greater, Val lesser) {
		if (lesser.baseRepresentation.compareTo(greater.baseRepresentation) > 0) {
			greater.baseRepresentation = lesser.baseRepresentation;
			greater.rawClass = lesser.rawClass;
		} else if (lesser.baseRepresentation.compareTo(greater.baseRepresentation) == 0){
			if (WalaNullPointerAnalysis.cha.isSubclassOf(greater.rawClass, lesser.rawClass)){
				greater.rawClass = lesser.rawClass;
			} else if (WalaNullPointerAnalysis.cha.isSubclassOf(lesser.rawClass, greater.rawClass)){
				System.out.println("ERROR: shouldn't be in this case. We assumed (greater<lesser)");
				System.exit(0);
			} else { // Neither is a subclass of the other
				greater.rawClass = WalaNullPointerAnalysis.cha.getLeastCommonSuperclass(greater.rawClass, lesser.rawClass);
			}
		} else {
			System.out.println("ERROR: shouldn't be in this case. We assumed (greater<lesser)");
			System.exit(0);
		}
		
	}
}
