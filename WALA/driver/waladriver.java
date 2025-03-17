import com.ibm.wala.classLoader.AnalysisScope;
import com.ibm.wala.classLoader.ClassLoaderFactory;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.AnalysisScopeReader;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.CallGraphBuilder;
import com.ibm.wala.ipa.callgraph.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.impl.Util;
import com.ibm.wala.ipa.cha.ClassHierarchy;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.config.AnalysisScopeReader;
import com.ibm.wala.util.io.FileProvider;

import java.io.File;
import java.io.IOException;

public class SimpleWalaDriver {
    public static void main(String[] args) throws IOException, ClassHierarchyException {
        if (args.length < 1) {
            System.err.println("Usage: java SimpleWalaDriver <path-to-jar>");
            System.exit(1);
        }

        String jarPath = args[0];
        
        // Create an analysis scope
        AnalysisScope scope = AnalysisScopeReader.makeJavaBinaryAnalysisScope(jarPath, new FileProvider().getFile("primordial.txt"));
        
        // Build the class hierarchy
        ClassHierarchy cha = ClassHierarchy.make(scope);
        
        // Set up call graph analysis options
        AnalysisOptions options = new AnalysisOptions();
        
        // Build the call graph
        CallGraphBuilder<?> builder = Util.makeZeroCFABuilder(options, scope, cha, null, null);
        CallGraph cg = builder.makeCallGraph(options);
        
        // Print call graph nodes
        cg.stream().forEach(node -> System.out.println(node));
    }
}
