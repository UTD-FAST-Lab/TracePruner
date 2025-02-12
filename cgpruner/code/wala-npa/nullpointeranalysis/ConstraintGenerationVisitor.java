package nullpointeranalysis;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ssa.*;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.intset.IntIterator;

public class ConstraintGenerationVisitor extends SSAInstruction.Visitor{
	private IClass currentClass;
	private IMethod currentMethod;
	private CGNode cgnode;
	private HashMap<Integer, Integer> nextMap;
	
	public ConstraintGenerationVisitor(CGNode n, HashMap<Integer, Integer> h) {
		cgnode = n;
		currentMethod = n.getMethod();
		currentClass = currentMethod.getDeclaringClass();
		nextMap = h;
	}
	/* VISIT FUNCTIONS */
    @Override
    public void visitArrayLength(SSAArrayLengthInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }
    
    @Override
    public void visitArrayLoad(SSAArrayLoadInstruction ins) {
    	simpleAssignment(ins, ins.getDef(), ins.getArrayRef() );
    }    
    
    @Override
    public void visitArrayStore(SSAArrayStoreInstruction ins) {
    	simpleAssignment(ins, ins.getArrayRef(), ins.getValue() );
    }   
    
    @Override
    public void visitBinaryOp(SSABinaryOpInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }   
    
    @Override
    public void visitCheckCast(SSACheckCastInstruction ins) {
    	simpleAssignment(ins, ins.getResult(), ins.getVal() );
    }    
    
    @Override
    public void visitComparison(SSAComparisonInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }   
    
    @Override
    public void visitConditionalBranch(SSAConditionalBranchInstruction ins) {
    	//System.out.println("CONDITIONAL:" + currentMethod.getReference().getSignature() + "$$" + ins.getTarget());
    	// Copy the L and T array values to the target
    	copyAllFieldsToTargetIns(ins, getTargetOrClosest(ins.getTarget(), cgnode));
    	// Also, copy the L and T array values to the next instruction.
    	copyFieldsToNextStatement(ins);
    }   
    
    @Override
    public void visitConversion(SSAConversionInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }    
    
    @Override
    public void visitGet(SSAGetInstruction ins) {
    	// The assigned local variable gets its value from the field.
    	Val lesserValElement = Utils.getOrCreateVal(DataStructure.H, ins.getDeclaredField().getSignature());
    	Val greaterValElement = Utils.getLElement(currentMethod, ins.getDef());
		Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(lesserValElement, greaterValElement));
    	// Copy the T array values to the next instruction
		copyFieldsToNextStatement(ins);
    }  
    
    @Override
    public void visitGetCaughtException(SSAGetCaughtExceptionInstruction ins) {
    	// Do nothing. It's as if this instruction doesn't exist, since it
    	// doesn't feature in the normal instruction list, and nothing
    	// interesting happens here.
    }  
    
    @Override
    public void visitGoto(SSAGotoInstruction ins) {
    	// Copy the T array values to the next instruction
    	copyAllFieldsToTargetIns(ins, getTargetOrClosest(ins.getTarget(), cgnode));
    }  
 
    @Override
    public void visitInstanceof(SSAInstanceofInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }    
    
    @Override
    public void visitInvoke(SSAInvokeInstruction ins) {
    	// Broken down into 3 parts, one each of the L, T and M arrays.
    	handleLArrayForInvoke(ins);
    	handleTArrayForInvoke(ins);
    	if (isInitCall(ins)) {
    		handleMArrayForInitInvoke(ins);
    	} else {
    		handleMArrayForNonInitInvoke(ins);
    	}
    }
    
	@Override
    public void visitLoadMetadata(SSALoadMetadataInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }
    
    @Override
    public void visitMonitor(SSAMonitorInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }  
 
    @Override
    public void visitNew(SSANewInstruction ins) {
    	Val lhsElement = Utils.getLElement(currentMethod, ins.getDef());
    	lhsElement.baseRepresentation = ValBase.NOTNULL;
    	// Copy the T array values to the next instruction
    	copyFieldsToNextStatement(ins);
    }
    
    @Override
    public void visitPhi(SSAPhiInstruction ins) {
    	for (int i = 0; i < ins.getNumberOfUses(); i++) {
    		Val lesserValElement = Utils.getLElement(currentMethod, ins.getUse(i));
        	Val greaterValElement = Utils.getLElement(currentMethod, ins.getDef());
    		Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(lesserValElement, greaterValElement));
    	}
    	// Don't do anything with T values, since this is not a normal instruction
    }    
    
    @Override
    public void visitPi(SSAPiInstruction ins) {
    	// Do nothing. It's as if this instruction doesn't exist, since it
    	// doesn't feature in the normal instruction list, and nothing
    	// interesting happens here.
    }
    
    @Override
    public void visitPut(SSAPutInstruction ins) {	
    	// The rhs value flows to the field
    	Val lesserValElement = Utils.getLElement(currentMethod, ins.getVal());
    	Val greaterValElement = Utils.getOrCreateVal(DataStructure.H, ins.getDeclaredField().getSignature());
		Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(lesserValElement, greaterValElement));
		
		// All the T values get copied to the next line, except the field defined
		for (IField f : currentClass.getDeclaredInstanceFields()) {
			if (f.getReference() == ins.getDeclaredField()) {
				Def fieldSet = Utils.getTElement(currentMethod, ins.iIndex() + 1, f.getReference());
				fieldSet.base = DefBase.DEF;
			} else {
				makeSimpleDefConstraint(ins.iIndex(), f.getReference(), ins.iIndex() + 1, f.getReference());
			}
		}
    }    
    
    @Override
    public void visitReturn(SSAReturnInstruction ins) {
    	// Part 1: The returned variable is <= the possible return values for the function.
    	MethodInfo minfoObject = Utils.getOrCreateMethodInfo(DataStructure.M, currentMethod);
    	if (ins.getResult() != Constants.voidType) {
    		Val lesserValElement = Utils.getLElement(currentMethod, ins.getResult());
    		Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(lesserValElement, minfoObject.returnVal));
    	}
    	
		
		// Part 2: Add to 'post' for each method
		for (IField f : currentClass.getDeclaredInstanceFields()) {
			Def lesserDefElement = Utils.getTElement(currentMethod, ins.iIndex(), f.getReference());
    		Def greaterDefElement = Utils.getOrCreateDef(minfoObject.post, f.getReference().getSignature());
    		Constraint.defSimpleConstraints.add(new SimpleConstraint<Def>(lesserDefElement, greaterDefElement));
		}
		
		// Part 3: If inside a constructor and a field is uninitialized,
		// mark it as MaybeNull
		if (currentMethod.isInit()) {
			for (IField f : currentClass.getDeclaredInstanceFields()) {
				Def defElementToCheck = Utils.getTElement(currentMethod, ins.iIndex(), f.getReference());
	    		Def undef = new Def(DefBase.UNDEF, "UNDEF-Constant");
	    		Val rhs = new Val(ValBase.MAYBENULL, null, "MAYBENULL-Constant");
	    		Val lhs = Utils.getOrCreateVal(DataStructure.H, f.getReference().getSignature());
	    		ConditionalAssignmentConstraint cc = new ConditionalAssignmentConstraint(defElementToCheck ,undef , lhs, rhs);
				Constraint.conditionalAssgnConstraints.add(cc);
			}
		}
    }
    
    @Override
    public void visitSwitch(SSASwitchInstruction ins) {
    	IntIterator labelIterator = ins.iterateLabels();
    	while(labelIterator.hasNext()) {
    		int target = ins.getTarget(labelIterator.next());
    		// Copy the T array values to the target
        	copyAllFieldsToTargetIns(ins, getTargetOrClosest(target, cgnode));
    	}
    	// Copy the T at array values to the default
    	copyAllFieldsToTargetIns(ins, getTargetOrClosest(ins.getDefault(), cgnode));
    }
    
    @Override
    public void visitThrow(SSAThrowInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }
    
    @Override
    public void visitUnaryOp(SSAUnaryOpInstruction ins) {
    	copyFieldsToNextStatement(ins);
    }
    
    // This will generate general program-level constraints.
    public static void generateGeneralConstraints() {
    	// Part 1: Set constraints for override pairs
    	ArrayList<Pair> overridePairs = getOverridePairs();
    	for (Pair p : overridePairs) {
    		MethodInfo minfoFirst = Utils.getOrCreateMethodInfo(DataStructure.M, p.first);
    		MethodInfo minfoSecond = Utils.getOrCreateMethodInfo(DataStructure.M, p.second);
    		if (minfoFirst.args.size() != minfoSecond.args.size()){
                continue;
            }
            for (int i = 0 ; i < minfoFirst.args.size() ; i++) {
    			Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(minfoFirst.args.get(i),minfoSecond.args.get(i)));
    		}
    		for (IField f : p.first.getDeclaringClass().getDeclaredInstanceFields()) {
    			Def d1 = Utils.getOrCreateDef(minfoFirst.post, f.getReference().getSignature());
    			d1.base = DefBase.UNDEF;
    		}
    		for (IField f : p.second.getDeclaringClass().getDeclaredInstanceFields()) {
    			Def d2 = Utils.getOrCreateDef(minfoSecond.post, f.getReference().getSignature());
    			d2.base = DefBase.UNDEF;
    		}
    	}
    	
    	// Part 2: Set constraints for the beginning of a method
    	for (CGNode cgnode : WalaNullPointerAnalysis.callgraph) {
    		if (cgnode.getIR() == null || cgnode.getIR().getInstructions() == null) {
    			continue; //skip these empty methods
    		}
    		IMethod m = cgnode.getMethod();
    		IClass c = m.getDeclaringClass();
			if (Constants.useOnlyTestClasess ) {
                if (!c.getName().getPackage().toString().startsWith("tests")){ //skip during debugging
                	continue; 
                }
            }
			MethodInfo minfo = Utils.getOrCreateMethodInfo(DataStructure.M, m);
			for (IField f : c.getDeclaredInstanceFields()) {
				Def lesserDefElement = Utils.getOrCreateDef(minfo.thisInitialization, f.getReference().getSignature());
				int firstInstructionIdx = getTargetOrClosest(0, cgnode);
				Def greaterDefElement = Utils.getTElement(m, firstInstructionIdx, f.getReference());
				Constraint.defSimpleConstraints.add(new SimpleConstraint<Def>(lesserDefElement, greaterDefElement));
			}
			for (int i = 0 ; i < minfo.args.size() ; i++) {
				Val greaterValElement = Utils.getLElement(m, i+1); // args get a valueNumber of +1 of their index
				Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(minfo.args.get(i),greaterValElement));
			}
    	}
    	
    	// Part 3: Set constraint for main method start.
    	for (CGNode entrypoint : WalaNullPointerAnalysis.callgraph.getEntrypointNodes()) {
    		IMethod entryMethod = entrypoint.getMethod();
    		for (IField f : entryMethod.getDeclaringClass().getDeclaredInstanceFields()) {
    			int firstInstructionIdx = getTargetOrClosest(0, entrypoint);
    			Def d3 = Utils.getTElement(entryMethod, firstInstructionIdx, f.getReference());
    			d3.base = DefBase.UNDEF;
    		}
    	}
    }
    
	/* HELPER FUNCTIONS */
    // Get pairs of methods which override each other
    private static ArrayList<Pair> getOverridePairs() {
    	ArrayList<Pair> overridePairs = new ArrayList<Pair>();
		for (IClass c1 : WalaNullPointerAnalysis.cha) {
			if (Constants.useOnlyTestClasess ) {
                if (!c1.getName().getPackage().toString().startsWith("tests")){ //skip during debugging
                	continue; 
                }
            }
			for (IClass c2 : WalaNullPointerAnalysis.cha.computeSubClasses(c1.getReference())) {
				if (c1 == c2) {
					continue;
				}
				// c2 is a proper subclass of c1. See if any pairs of methods have the
				// same name and signature
				for (IMethod m1 : c1.getDeclaredMethods()) {
					//for (IMethod m2 : c2.getDeclaredMethods()) {
					for (IMethod m2 : WalaNullPointerAnalysis.cha.getPossibleTargets(c2, m1.getReference())) {
						if (m2.getDeclaringClass()==c2) {
							// If they have the same name and signature
							if (m1.getSelector().toString().equals(m2.getSelector().toString())) {
								overridePairs.add(new Pair(m1,m2));
							}
						}
					}
				}
			}
		}
		return overridePairs;
	}
    
    void copyFieldsToNextStatement(SSAInstruction ins) {
    	// Copy the T array values to the next instruction
    	int nextInsIndex = nextMap.get(ins.iIndex());
    	if (nextInsIndex != Constants.noIndex) {
    		copyAllFieldsToTargetIns(ins, nextInsIndex);
    	}
    }
    
    // This copies the 'T' value (initialization) of each field to the
    // target instruction, for all fields in the class.
    void copyAllFieldsToTargetIns(SSAInstruction ins, int targetIns) {
    	for (IField f : currentClass.getDeclaredInstanceFields()) {
    		makeSimpleDefConstraint(ins.iIndex(), f.getReference(), targetIns, f.getReference());
    	}
    }
    
    // For a simple assignment type statement of the form "lhsValueNumber = rhsValueNumber"
    void simpleAssignment(SSAInstruction ins, int lhsValueNumber, int rhsValueNumber) {
    	Val lesserValElement = Utils.getLElement(currentMethod, rhsValueNumber);
    	Val greaterValElement = Utils.getLElement(currentMethod, lhsValueNumber);
		Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(lesserValElement, greaterValElement));
    	// Copy the T array values to the next instruction
		copyFieldsToNextStatement(ins);
    }
    
    // Creates a simple Def constraint and adds to the list of constraints.
    // The constraint simple looks like 
    // T[currentMethod,srcLine, srcField] <= L[currentMethod,targetLine,destField]
    void makeSimpleDefConstraint(int srcLine, FieldReference srcField, int targetLine, FieldReference targetField) {
    	Def lesserDefElement = Utils.getTElement(currentMethod, srcLine, srcField);
		Def greaterDefElement = Utils.getTElement(currentMethod, targetLine, targetField);
		Constraint.defSimpleConstraints.add(new SimpleConstraint<Def>(lesserDefElement, greaterDefElement));
    }
    
    private void handleMArrayForInitInvoke(SSAInvokeInstruction ins) {
		IMethod target = getMonomorphicTarget(ins);
		MethodInfo minfo = Utils.getOrCreateMethodInfo(DataStructure.M, target);
		// Part1: Compute constraint for 'thisInitialization' array
		for (IField f : target.getDeclaringClass().getDeclaredInstanceFields()) {
			Def fieldInitialization = Utils.getOrCreateDef(minfo.thisInitialization, f.getReference().getSignature());
			fieldInitialization.base = DefBase.UNDEF;
		}
		// Part2: Compute constraints for args
		minfo.args.get(0).baseRepresentation = ValBase.RAWT;
		for (int i = 1 ; i < minfo.args.size() ; i++) {
			Val lesserValElement = Utils.getLElement(currentMethod, ins.getUse(i));
			Val greaterValElement = minfo.args.get(i);
			Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(lesserValElement, greaterValElement));
		}
	}
    
	private void handleMArrayForNonInitInvoke(SSAInvokeInstruction ins) {
		// Part1: Compute constraint for 'thisInitialization' array
		boolean singleTargetAndCurrentClass = false;
		Set<CGNode> targets = WalaNullPointerAnalysis.getTargets(cgnode, ins.getCallSite());
		if (targets.size() == 1) {
			IMethod targetMethod = getMonomorphicTarget(ins);
			MethodInfo minfo = Utils.getOrCreateMethodInfo(DataStructure.M, targetMethod);
			if (targetMethod.getDeclaringClass() == currentClass) {
				singleTargetAndCurrentClass = true;
				for (IField f : currentClass.getDeclaredInstanceFields()) {
					Def lesserDefElement = Utils.getTElement(currentMethod, ins.iIndex(), f.getReference());
					Def greaterDefElement = Utils.getOrCreateDef(minfo.thisInitialization, f.getReference().getSignature());
					Constraint.defSimpleConstraints.add(new SimpleConstraint<Def>(lesserDefElement, greaterDefElement));
				}
			}
		}
		if (!singleTargetAndCurrentClass && !ins.isStatic()) {
			for (CGNode target : targets) {
				IMethod targetMethod = target.getMethod();
				MethodInfo minfo = Utils.getOrCreateMethodInfo(DataStructure.M, targetMethod);
				for (IField f : targetMethod.getDeclaringClass().getDeclaredInstanceFields()) {
					Val checkLesser = new Val(ValBase.RAW, targetMethod.getDeclaringClass(), "RAW(" + targetMethod.getDeclaringClass() + ")-Constant");
					Val checkGreater = Utils.getLElement(currentMethod, ins.getReceiver());
					Def lhs = Utils.getOrCreateDef(minfo.thisInitialization, f.getReference().getSignature()); 
					Def rhs = new Def(DefBase.UNDEF, "UNDEF-Constant");
					Constraint.type2conditionalAssgnConstraints.add(new ConditionalAssignmentConstraintType2(checkLesser, checkGreater, lhs, rhs));
				}
			}
		}
		
		// Part2: Compute constraints for args
		for (CGNode target : targets) {
			IMethod targetMethod = target.getMethod();
			MethodInfo minfo = Utils.getOrCreateMethodInfo(DataStructure.M, targetMethod);
			int firstArgIndex; // index of the first argument which is not 'this'
			if (targetMethod.isStatic()) {
				firstArgIndex = 0;
			} else {
				firstArgIndex = 1; // since arg0 is the 'this' variable
				// Add the constraint for arg0 ('this' variable)
				// The constraint is different from the rest of the arguments because
				// the value should flow only if the receiver-object is not MaybeNull (since 'this'
				// cannot be null)
				Val elementToCheck = Utils.getLElement(currentMethod, ins.getUse(0));
				Val checkAgainst = new Val(ValBase.MAYBENULL, null, "MAYBENULL-Constant");
				Val lesserValElement = Utils.getLElement(currentMethod, ins.getUse(0));
				Val greaterValElement = minfo.args.get(0);
				Constraint.antiConditionalConstraints.add(new AntiConditionalConstraint(elementToCheck, checkAgainst, lesserValElement, greaterValElement));
			}
			for (int i = firstArgIndex ; i < minfo.args.size() ; i++) {
				Val lesserValElement = Utils.getLElement(currentMethod, ins.getUse(i));
				Val greaterValElement = minfo.args.get(i);
				Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(lesserValElement, greaterValElement));
			}
		}
		
	}
	
	private void handleTArrayForInvoke(SSAInvokeInstruction ins) {
		// This boolean is set to true if the target size is one and the
		// class of the target is the current class.
		boolean conditionalConstaintUsed = false;
		Set<CGNode> targets = WalaNullPointerAnalysis.getTargets(cgnode, ins.getCallSite());
		if (targets.size() == 1) {
			IMethod target = getMonomorphicTarget(ins);
			if (target.getDeclaringClass() == currentClass) {
				conditionalConstaintUsed = true;
				for (IField f : currentClass.getDeclaredInstanceFields()) {
					MethodInfo minfo = Utils.getOrCreateMethodInfo(DataStructure.M, target);
					Def valueToCheck = Utils.getOrCreateDef(minfo.post, f.getReference().getSignature());
					Def expectedValue = new Def(DefBase.UNDEF, "UNDEF-Constant");
					Def lesserElement = Utils.getTElement(currentMethod, ins.iIndex(), f.getReference());
					Def greaterElement = Utils.getTElement(currentMethod, ins.iIndex()+1, f.getReference());
					Constraint.conditionalConstraints.add(new ConditionalConstraint(valueToCheck, expectedValue, lesserElement, greaterElement));
				}
			}
		}
		if (!conditionalConstaintUsed) {
			copyFieldsToNextStatement(ins);
		}
	}
	
	private void handleLArrayForInvoke(SSAInvokeInstruction ins) {
		// Part1: Compute L-value for receiver object
		if (!ins.isStatic()){ // static calls have no receiver object
			if (isInitCall(ins)) {
				Val greaterValElement = Utils.getLElement(currentMethod, ins.getReceiver());
				IClass c = getMonomorphicTarget(ins).getDeclaringClass();
				Val lesserValElement = new Val(ValBase.RAW, c, "RAW(" + c.getName().toString() + ")-CONSTANT");
				Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(lesserValElement,greaterValElement));
			} 
		}
		
		// Part2: Compute L-value for the lhs variable
		if (ins.getNumberOfDefs() != 0) { //for non-void return type.
			for (CGNode targetnode : WalaNullPointerAnalysis.getTargets(cgnode, ins.getCallSite())){
				if (!targetnode.getMethod().isInit() && ins.getDef() != Constants.voidType) {
					MethodInfo mInfo = Utils.getOrCreateMethodInfo(DataStructure.M, targetnode.getMethod());
			    	Val greaterValElement = Utils.getLElement(currentMethod, ins.getDef());
					Constraint.valSimpleConstraints.add(new SimpleConstraint<Val>(mInfo.returnVal, greaterValElement));
				}				
			}
		}
	}
	
	// Must make sure that the call is monomorphic, otherwise it will return
	// null;
	private IMethod getMonomorphicTarget(SSAInvokeInstruction ins) {
		Set<CGNode> targetnodes = WalaNullPointerAnalysis.getTargets(cgnode, ins.getCallSite());
		if (targetnodes.size()!=1) {
			System.out.println("getClassOfMonomorphicCall called with non-monomorphic call");
			System.exit(0); 
		}
		CGNode node = new ArrayList<CGNode>(targetnodes).get(0);
		return node.getMethod();
	}
	
	// Tells you whether the invoke instruction is for an init (constructor) method.
	private boolean isInitCall(SSAInvokeInstruction ins) {
		Set<CGNode> targetnodes = WalaNullPointerAnalysis.getTargets(cgnode, ins.getCallSite());
		if (targetnodes.size()!=1) {
			return false; // can't be an init call if it is not monomorphic
		}
		CGNode node = new ArrayList<CGNode>(targetnodes).get(0);
		if (node.getMethod().isInit()) {
			return true;
		} else {
			return false;
		}
	}
	
	// Returns the lowest instruction index greater than or equal to iIndex
	// which is not null;
	private static int getTargetOrClosest(int iIndex, CGNode cgnode) {
		SSAInstruction[] insArray = cgnode.getIR().getInstructions();
		for (int i = iIndex ; i < insArray.length ; i++) {
			if (insArray[i]!=null) {
				return i;
			}
		}
		return -1;
	}
}