package nullpointeranalysis;


import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Properties;
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

public class PhiInstructionPrinter {
	public static ClassHierarchy cha; 
	public static CallGraph callgraph;
	
    public static void main(String[] args) throws WalaException, IllegalArgumentException, CancelException, IOException {
        Properties p = CommandLine.parse(args);
        String classpath = p.getProperty("jar");
        String mainclass = p.getProperty("mainclass");
        
        // Build the callgraph
        AnalysisScope scope = AnalysisScopeReader.makeJavaBinaryAnalysisScope(classpath, null);
        cha = ClassHierarchyFactory.make(scope);
        Iterable<Entrypoint> entrypoints = Util.makeMainEntrypoints(scope, cha, "L" + mainclass.replaceAll("\\.","/"));
        AnalysisOptions options = new AnalysisOptions(scope, entrypoints);
        CallGraphBuilder<?> builder = Util.makeZeroCFABuilder(Language.JAVA, options, new AnalysisCacheImpl(), cha, scope);;     
        callgraph = builder.makeCallGraph(options, null);
        
        // Print out all instructions
        for(Iterator<CGNode> it = callgraph.iterator(); it.hasNext(); ) {
            CGNode cgnode = it.next();
            IR ir = cgnode.getIR();    
            if (ir!=null) {
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
    }
}