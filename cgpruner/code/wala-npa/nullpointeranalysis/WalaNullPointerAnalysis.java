package nullpointeranalysis;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;

import com.ibm.wala.ipa.callgraph.*;
import com.ibm.wala.ipa.callgraph.impl.Util;
import com.ibm.wala.ipa.cha.ClassHierarchy;
import com.ibm.wala.ipa.cha.ClassHierarchyFactory;

import com.ibm.wala.classLoader.*;
import com.ibm.wala.types.*;
import com.ibm.wala.util.*;
import com.ibm.wala.util.config.AnalysisScopeReader;
import com.ibm.wala.util.io.CommandLine;
import com.ibm.wala.ssa.*;
import com.ibm.wala.ipa.callgraph.AnalysisOptions.ReflectionOptions;

public class WalaNullPointerAnalysis {
	public static ClassHierarchy cha;
	public static CallGraph callgraph;
	public static boolean usePrunedCallgraph;
	public static HashSet<String> prunedCallgraphEdgeSet;
	public static HashSet<String> prunedCallgraphNodeSet;

	public static void main(String[] args)
			throws WalaException, IllegalArgumentException, CancelException, IOException {
		Properties p = CommandLine.parse(args);
		String classpath = p.getProperty("classpath");
		String mainclass = p.getProperty("mainclass");
		String prunedCallgraphFile = p.getProperty("prunedCallgraph");
		// Check if we are using a pruned-callgraph. If yes, populate it
		if (prunedCallgraphFile != null) {
			usePrunedCallgraph = true;
			readPrunedCallgraph(prunedCallgraphFile);
		} else {
			usePrunedCallgraph = false;
		}
		// Boilerplate code for making callgraph
		AnalysisScope scope = AnalysisScopeReader.makeJavaBinaryAnalysisScope(classpath, null);
		cha = ClassHierarchyFactory.make(scope);
		Iterable<Entrypoint> entrypoints = Util.makeMainEntrypoints(scope, cha, "L" + mainclass.replaceAll("\\.", "/"));
		AnalysisOptions options = new AnalysisOptions(scope, entrypoints);
		options.setReflectionOptions(ReflectionOptions.NONE);
    if (Constants.debug)
			System.out.println(">>>>>Call-graph start");
		CallGraphBuilder<?> builder = Util.makeZeroCFABuilder(Language.JAVA, options, new AnalysisCacheImpl(), cha,scope);
		callgraph = builder.makeCallGraph(options, null);
		// Finally, run the Null Pointer Analysis
		runNPA();
		System.out.println("Done");
	}

	public static void runNPA() {
		runConstraintGeneration();

		if (Constants.debug)
			System.out.println(">>>>>Start-NPA-Constraint-Solver");
		ConstraintSolver.solve();

		if (Constants.debug)
			System.out.println(">>>>>Start-Null-Dereference-Flagging");
		flagNullDereferences();
		
		for (String nullWarning : NullDerferenceChecker.nullWarnings) {
			System.out.println(nullWarning);
		}
		//System.out.println(NullDerferenceChecker.nullWarnings.size());
	}

	/* HELPER functions */

	private static void runConstraintGeneration() {
		if (Constants.debug)
			System.out.println(">>>>>Start-NPA-constraint-generation-visitor");
		for (Iterator<CGNode> it = callgraph.iterator(); it.hasNext();) {
			CGNode cgnode = it.next();
			if (Constants.useOnlyTestClasess) {
				TypeName t1 = cgnode.getMethod().getDeclaringClass().getName();
				if (!t1.getPackage().toString().startsWith("tests")) { // skip during debugging
					continue;
				}
			}
			if (Constants.debug) {
				System.out.println("\n>>>>>Method-sign:" + cgnode.getMethod().getReference().getSignature());
				printInstructions(cgnode);
			}
			if (usePrunedCallgraph && isStdLib(cgnode)) { //Skip the std. lib if we are using pruned callgraph
				continue;
			}
			/*if (usePrunedCallgraph) {
				String nodeString = StringFormattingUtils.getMethodString(cgnode);
				if (!prunedCallgraphNodeSet.contains(nodeString)) {
					continue; // skip nodes which are not in the pruned call-graph
				}
			}*/
			IR ir = cgnode.getIR();
			if (ir != null) {
				ir.visitAllInstructions(new ConstraintGenerationVisitor(cgnode, getNextInstructionMapping(cgnode)));
			}
		}
		ConstraintGenerationVisitor.generateGeneralConstraints();
	}

	private static void flagNullDereferences() {
		// Flagging possible null dereferences based on constraint-solver solution.
		for (Iterator<CGNode> it = callgraph.iterator(); it.hasNext();) {
			CGNode cgnode = it.next();
			if (Constants.useOnlyTestClasess) {
				TypeName t1 = cgnode.getMethod().getDeclaringClass().getName();
				if (!t1.getPackage().toString().startsWith("tests")) { // skip during debugging
					continue;
				}
			}
			if (!Constants.reportNullsInStdLib && isStdLib(cgnode)) {
				continue;
			}
			IR ir = cgnode.getIR();
			if (ir != null) {
				ir.visitAllInstructions(new NullDerferenceChecker(cgnode));
			}
		}
	}

	private static void readPrunedCallgraph(String prunedCallgraphFile) throws FileNotFoundException {
		// Read Callgraph edge set
		prunedCallgraphEdgeSet = new HashSet<String>();
		File pcgFile = new File(prunedCallgraphFile);
		Scanner reader = new Scanner(pcgFile);
		while (reader.hasNextLine()) {
			prunedCallgraphEdgeSet.add(reader.nextLine());
		}
		reader.close();
	}

	private static void printInstructions(CGNode cgnode) {
		if (cgnode.getIR() != null) {
			Iterator<SSAInstruction> insIterator = cgnode.getIR().iterateAllInstructions();
			while (insIterator.hasNext()) {
				SSAInstruction ins = insIterator.next();
				if (ins != null) {
					System.out.println("index:" + ins.iIndex());
					System.out.println(ins);
				}
			}
		}
	}

	public static Set<CGNode> getTargets(CGNode srcMethod, CallSiteReference callsite) {
		if (usePrunedCallgraph) {
			// Use only those edges which are present in the pruned-callgraph
			HashSet<CGNode> targetSet = new HashSet<CGNode>();
			for (CGNode dest : callgraph.getPossibleTargets(srcMethod, callsite)) {
				if (isStdLib(dest)) {
					continue;
				}
				String edgeSign = StringFormattingUtils.getEdgeSignature(srcMethod, callsite, dest);
				if (prunedCallgraphEdgeSet.contains(edgeSign)) {
					targetSet.add(dest);
				}
			}
			return targetSet;
		} else {
			// Just return the normal targets
			return callgraph.getPossibleTargets(srcMethod, callsite);
		}
	}

	private static HashMap<Integer, Integer> getNextInstructionMapping(CGNode cgnode) {
		HashMap<Integer, Integer> nextMap = new HashMap<Integer, Integer>();
		// Then add all the local variables.
		SSAInstruction[] instructionList = cgnode.getIR().getInstructions();
		for (int i = 0; i < instructionList.length; i++) {
			if (instructionList[i] != null) {
				// find the next non-null instruction
				boolean foundNext = false;
				for (int j = i + 1; j < instructionList.length; j++) {
					if (instructionList[j] != null) {
						nextMap.put(i, j);
						foundNext = true;
						break;
					}
				}
				if (!foundNext) {
					nextMap.put(i, -1); // corner case when there is no next instruction
				}
			}
		} // the last instruction must be a return statement anyways.
		return nextMap;
	}

	private static boolean isStdLib(CGNode cgnode) {
		TypeName t1 = cgnode.getMethod().getDeclaringClass().getName();
		if (t1.getPackage() == null){
        return false;
    }
    String packageName = t1.getPackage().toString();
		if (packageName.startsWith("java/") ||
			packageName.startsWith("javax/") ||
		    packageName.startsWith("sun/") || 
		    packageName.startsWith("com/oracle/")  || 
		    packageName.startsWith("com/sun/")  || 
		    packageName.startsWith("org/ietf/")) { // skip during debugging
			return true;
		}
		return false;
	}
	/* UNUSED HELPER FUNCTIONS */

	/*
	 * // Returns the set of valueNumbers used in the method // Achieves this by
	 * adding all the Uses and Defs in all the instructions. private static
	 * Set<Integer> getValueNumbersInMethod(CGNode cgnode){ HashSet<Integer>
	 * valueNums = new HashSet<Integer>(); // First add all the formal parameters
	 * for (int i = 1 ; i <= cgnode.getMethod().getNumberOfParameters() ; i++) {
	 * valueNums.add(i); } // Then add all the local variables.
	 * Iterator<SSAInstruction> insIterator =
	 * cgnode.getIR().iterateAllInstructions(); while (insIterator.hasNext()) {
	 * SSAInstruction ins = insIterator.next(); if (ins != null) {
	 * System.out.println("index:" + ins.iIndex()); System.out.println(ins); for
	 * (int i=0; i<ins.getNumberOfDefs(); i++) { valueNums.add(ins.getDef(i)); } for
	 * (int i=0; i<ins.getNumberOfUses(); i++) { valueNums.add(ins.getUse(i)); } } }
	 * return valueNums; }
	 * 
	 * private static boolean isStdLib(TypeName t1){ if (t1!=null){ String pName =
	 * t1.getPackage().toString(); if (pName!=null){ if (pName.startsWith("java/")
	 * || pName.startsWith("javax/") || pName.startsWith("sun/") ||
	 * pName.startsWith("com/ibm/wala")){ return false; } else {
	 * System.out.println("---" + pName); return true; } } } return true; }
	 */
}
