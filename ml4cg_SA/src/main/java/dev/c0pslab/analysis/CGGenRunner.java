package dev.c0pslab.analysis;

import dev.c0pslab.analysis.cg.CGBuilder;
import dev.c0pslab.analysis.cg.CallGraphBuilders;
import dev.c0pslab.analysis.cg.GlobalConstants;
import dev.c0pslab.analysis.cg.alg.*;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ibm.wala.ipa.callgraph.AnalysisOptions.ReflectionOptions;

import picocli.CommandLine;
import picocli.CommandLine.Option;

import java.io.IOException;

public class CGGenRunner implements Runnable {
    private static final Logger LOG = LoggerFactory.getLogger(CGGenRunner.class);

    @CommandLine.Option(names = {"-o", "--output"}, description = "An output path with a file name to save source" +
            " to target candidates")
    String outputJSONFile;

    @CommandLine.Option(names = {"-j", "--jars"}, description = "A CSV file containing list of project names and " +
            "their respective JAR files to be analyzed")
    String inputJars;

    // CG algorithms
    // @CommandLine.Option(names = {"-0cfa", "--zerocfa"}, defaultValue = "false" ,description = "Build a CG with the 0-CFA algorithm")
    // boolean useZeroCFA;

    // @CommandLine.Option(names = {"-1cfa", "--onecfa"}, defaultValue = "false" ,description = "Build a CG with the 1-CFA algorithm")
    // boolean useOneCFA;

    @CommandLine.Option(names = {"-p", "--parallel"}, defaultValue = "false" ,description = "Build a CG with the RTA algorithm")
    boolean useParallelism;

    @CommandLine.Option(names = {"-t", "--threads"}, defaultValue = "1" ,description = "Number of threads for building call graphs",
            type = int.class)
    int numberOfThreads;

    // my code from here

    @CommandLine.Option(
            names = "--reflectionSetting",
            description = "Valid values: ${COMPLETION-CANDIDATES}",
            defaultValue = "NONE")
    ReflectionOptions reflection;

    @CommandLine.Option(
            names = "--handleStaticInit",
            description =
                    "Should call graph construction handle "
                            + "possible invocations of static initializer methods?")
    boolean handleStaticInit;

    @CommandLine.Option(
            names = "--useConstantSpecificKeys",
            description = "Use distinct instance keys for " + "distinct string constants?")
    boolean useConstantSpecificKeys;

    @CommandLine.Option(
            names = "--useStacksForLexcialScoping",
            description = "Should analysis of lexical " + "scoping consider call stacks?")
    boolean useStacksForLexicalScoping;

    @CommandLine.Option(
            names = "--useLexicalScopingForGlobals",
            description =
                    "Should global variables be " + "considered lexically-scoped from the root node?")
    boolean useLexicalScopingForGlobals;

    @CommandLine.Option(
            names = "--handleZeroLengthArray",
            description = "Should call graph construction handle " + "arrays of zero-length differently?",
            defaultValue = "true")
    boolean handleZeroLengthArray;

    @CommandLine.Option(
            names = "--cgalgo",
            description = "Valid values: ${COMPLETION-CANDIDATES}",
            defaultValue = "ZERO_CFA")
    CallGraphBuilders callGraphBuilder;

    @CommandLine.Option(
            names = "--sensitivity",
            description =
                    "Level of context/object sensitivity (only used if cg algo is NCFA, NOBJ, VANILLA_NCFA, or VANILLA_NOBJ)",
            defaultValue = "1")
    int sensitivity;

    @CommandLine.Option(
            names = "--confID",
            description =
                    "ID of the current configuration",
            defaultValue = "1")
    int confID;


    @Override
    public void run() {
        GlobalConstants.useParallelism = useParallelism;
        GlobalConstants.maxNumberOfThreads = numberOfThreads;
        LOG.info("Max Heap size: " + FileUtils.byteCountToDisplaySize(Runtime.getRuntime().maxMemory()));
        LOG.info("Number of threads to use: " + GlobalConstants.maxNumberOfThreads);
        final var startTime = System.nanoTime();
        try {
            // if (useZeroCFA) {
            //     LOG.info("Using the 0-CFA approach");
            //    buildCGsWithZeroCFA();
            // } else if (useOneCFA) {
            //     LOG.info("Using the 1-CFA approach");
            //     buildCGsWithOneCFA();
            // } else {
            //     throw new RuntimeException("Choose a CG algorithm!");
            // }
            buildCG();
            final var elapsedTimeInSeconds = (double)(System.nanoTime() - startTime) / 1_000_000_000.0;
            LOG.info("Elapsed Time: " + String.format("%.02f", elapsedTimeInSeconds) + " seconds");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new CGGenRunner()).execute(args));
    }

    private void buildCGsWithZeroCFA() throws IOException {
        ZeroCFA.buildCG(inputJars, outputJSONFile);
    }

    private void buildCGsWithOneCFA() throws IOException {
        OneCFA.buildCG(inputJars, outputJSONFile);
    }

    private void buildCG() throws IOException {
        CGBuilder.buildCG(inputJars, outputJSONFile, reflection, handleStaticInit, useConstantSpecificKeys, useStacksForLexicalScoping, useLexicalScopingForGlobals, handleZeroLengthArray, callGraphBuilder, sensitivity, confID);
    }
}
