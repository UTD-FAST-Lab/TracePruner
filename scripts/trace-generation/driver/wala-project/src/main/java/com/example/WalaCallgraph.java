package com.example;


import java.io.FileWriter;  
import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.Properties;
import java.util.Set;
import java.util.Arrays;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

import com.ibm.wala.ipa.callgraph.*;
import com.ibm.wala.ipa.callgraph.impl.Util;
import com.ibm.wala.ipa.cha.ClassHierarchy;
import com.ibm.wala.ipa.cha.ClassHierarchyFactory;

import com.ibm.wala.classLoader.*;
import com.ibm.wala.types.*;
import com.ibm.wala.util.*;
import com.ibm.wala.core.util.config.AnalysisScopeReader;
import com.ibm.wala.util.io.CommandLine;

public class WalaCallgraph {
  
  public static void main(String[] args) throws WalaException, IllegalArgumentException, CancelException, IOException {
    Properties p = CommandLine.parse(args);
    String classpath = p.getProperty("classpath");
    String mainclass = p.getProperty("mainclass");
    String outputfile = p.getProperty("output");
    String exclude = p.getProperty("exclude");
    String analysis = p.getProperty("analysis");
    String reflection = p.getProperty("reflection"); 
    String resolveinterfaces = p.getProperty("resolveinterfaces");
    //resolveinterfaces = false results in an analysis which does not resolve an interface edge to its actual possible targets

    String fileName = "wala-exclusion.txt"; // Example file in src/main/resources
    File exclusion = null;
    // Get the resource URL
    URL resourceUrl = WalaCallgraph.class.getClassLoader().getResource(fileName);
    if (resourceUrl != null) {
      exclusion = new File(resourceUrl.getFile());
    }

    AnalysisScope scope = AnalysisScopeReader.instance.makeJavaBinaryAnalysisScope(classpath, exclusion);
    ClassHierarchy cha = ClassHierarchyFactory.make(scope);

    Iterable<Entrypoint> entrypoints = Util.makeMainEntrypoints(scope, cha, "L" + mainclass.replaceAll("\\.","/"));
    AnalysisOptions options = new AnalysisOptions(scope, entrypoints);

    /* Choose the correct reflection option */
    if (reflection.equalsIgnoreCase("true")){
        options.setReflectionOptions(AnalysisOptions.ReflectionOptions.NO_FLOW_TO_CASTS_APPLICATION_GET_METHOD);
    } else {
        options.setReflectionOptions(AnalysisOptions.ReflectionOptions.NONE);
    }
    
    /* Choose the correct analysis option */
    CallGraphBuilder builder;
    switch(analysis) {
        case "0cfa":
            builder = Util.makeZeroCFABuilder(Language.JAVA, options, new AnalysisCacheImpl(), cha, scope);
            break;
        case "1cfa":
            builder = Util.makeNCFABuilder(1, options, new AnalysisCacheImpl(), cha, scope);
            break;
        case "rta":
            builder = Util.makeRTABuilder(options, new AnalysisCacheImpl(), cha, scope);
            break;
        default:
            System.out.println("-----Invalid analysis option----");
            builder = null;
    }
    
    CallGraph graph = builder.makeCallGraph(options, null);
  }
}