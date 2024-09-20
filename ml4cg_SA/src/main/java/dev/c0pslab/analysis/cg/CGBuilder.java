package dev.c0pslab.analysis.cg;


import com.ibm.wala.classLoader.Language;
import com.ibm.wala.ipa.callgraph.AnalysisCacheImpl;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.CallGraphBuilder;
import com.ibm.wala.ipa.callgraph.CallGraphBuilderCancelException;
import com.ibm.wala.ipa.callgraph.AnalysisOptions.ReflectionOptions;
import com.ibm.wala.ipa.callgraph.impl.Util;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchy;
import dev.c0pslab.analysis.cg.*;
import dev.c0pslab.analysis.cg.alg.OneCFA;
import dev.c0pslab.analysis.utils.CSVReader;
import dev.c0pslab.analysis.utils.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.*;

public class CGBuilder {
     private static final Logger LOG = LoggerFactory.getLogger(OneCFA.class);
    private static final int N = 1;

    static public void buildCG(final String inputJars, final String outputPath, final ReflectionOptions reflection, final boolean handleStaticInit, final boolean useConstantSpecificKeys, final boolean useStacksForLexicalScoping, final boolean useLexicalScopingForGlobals, final boolean handleZeroLengthArray, final CallGraphBuilders callGraphBuilder, final int sensitivity, final int confID) throws IOException {
        var trainProgramsJars = CSVReader.readProjectToJarCsv(inputJars);
        final var filteredTrainProgramsJars = trainProgramsJars;

        for (var programJar : filteredTrainProgramsJars.entrySet()) {
            final var programJarFile = Paths.get(programJar.getValue());
            final var programJarFileDeps = CGUtils.getProgramDependencies(programJarFile);
            final var programJarName = programJar.getKey();
            // programJarName.contains("/") ? programJarName.replace("/", "_") :
            final var programOutputCGFolder = Paths.get(outputPath, programJarName);
            final var programOutputCGFile = Paths.get(programOutputCGFolder.toString(), programJarFile.getFileName().toString().replace(".jar", ""));

            final var cha = new CHA();
            final ClassHierarchy chaMap;
            try {
                chaMap = cha.construct(programJarFile.toString(), programJarFileDeps,
                        CGUtils.createWalaExclusionFile().toFile());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            final var entryPoints = new EntryPointsGenerator(chaMap).getEntryPoints();
            final var opts = new AnalysisOptions(cha.getCHAScope(), entryPoints);
            final var cache = new AnalysisCacheImpl();
            // Util.makeNCFABuilder(N, opts, cache, chaMap);
            
            // set options
            opts.setReflectionOptions(reflection);
            opts.setHandleStaticInit(handleStaticInit);
            opts.setUseConstantSpecificKeys(useConstantSpecificKeys);
            opts.setUseStacksForLexicalScoping(useStacksForLexicalScoping);
            opts.setUseLexicalScopingForGlobals(useLexicalScopingForGlobals);
            opts.setHandleZeroLengthArray(handleZeroLengthArray);
            CallGraphBuilder<InstanceKey> cgBuilder;
            
            switch (callGraphBuilder) {
                case NCFA:
                    cgBuilder = Util.makeNCFABuilder(sensitivity, opts, cache, chaMap);
                    break;
                case NOBJ:
                    cgBuilder = Util.makeNObjBuilder(sensitivity, opts, cache, chaMap);
                    break;
                case VANILLA_NCFA:
                    cgBuilder =
                            Util.makeVanillaNCFABuilder(sensitivity, opts, cache, chaMap);
                    break;
                case VANILLA_NOBJ:
                    cgBuilder =
                            Util.makeVanillaNObjBuilder(sensitivity, opts, cache, chaMap);
                    break;
                case RTA:
                    cgBuilder = Util.makeRTABuilder(opts, cache, chaMap);
                    break;
                case ZERO_CFA:
                    cgBuilder = Util.makeZeroCFABuilder(Language.JAVA, opts, cache, chaMap);
                    break;
                case ZEROONE_CFA:
                    cgBuilder = Util.makeZeroOneCFABuilder(Language.JAVA, opts, cache, chaMap);
                    break;
                case VANILLA_ZEROONECFA:
                    cgBuilder =
                            Util.makeVanillaZeroOneCFABuilder(Language.JAVA, opts, cache, chaMap);
                    break;
                case ZEROONE_CONTAINER_CFA:
                    cgBuilder = Util.makeZeroOneContainerCFABuilder(opts, cache, chaMap);
                    break;
                case VANILLA_ZEROONE_CONTAINER_CFA:
                    cgBuilder = Util.makeVanillaZeroOneContainerCFABuilder(opts, cache, chaMap);
                    break;
                case ZERO_CONTAINER_CFA:
                    cgBuilder = Util.makeZeroContainerCFABuilder(opts, cache, chaMap);
                    break;
                default:
                    throw new IllegalArgumentException("Invalid call graph algorithm.");
            }
            
            
            
            // final SSAPropagationCallGraphBuilder cgBuilder = Util.makeZeroOneCFABuilder(Language.JAVA, opts, cache, chaMap);
            
            
            
            
            LOG.info("Building a call graph for " + programJarName);
            final var startTime = System.nanoTime();
            Callable<Void> task = () -> {
                final CallGraph cg;
                try {
                    cg = cgBuilder.makeCallGraph(opts, null);
                    double durationInSeconds = (System.nanoTime() - startTime) / 1_000_000_000.0;
                    LOG.info("Generated a call graph for " + programJarFile.getFileName() + " in " + durationInSeconds + " seconds");
                    Files.createDirectories(programOutputCGFolder);
                    IOUtils.writeWalaCGToFile(cg, programOutputCGFile.toString() + "_config" + String.valueOf(confID));
                } catch (CallGraphBuilderCancelException | IOException e) {
                    LOG.error("Failed to generate a call graph for " + programJarFile.getFileName());
                    e.printStackTrace();
                }
                return null;
            };
            ExecutorService executor = Executors.newSingleThreadExecutor();
            Future<Void> future = executor.submit(task);

            try {
                future.get(GlobalConstants.maxHoursToGenerateCG, TimeUnit.HOURS);
            } catch (TimeoutException e) {
                LOG.error("Failed to generate a call graph for " + programJarFile.getFileName() +" in the specified time");
                future.cancel(true);
            } catch (InterruptedException | ExecutionException e) {
                LOG.error("Failed to generate a call graph for " + programJarFile.getFileName());
            } finally {
                executor.shutdownNow();
            }
        }
    }
}
