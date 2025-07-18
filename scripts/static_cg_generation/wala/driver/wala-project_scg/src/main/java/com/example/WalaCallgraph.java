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

    //Lambda start string constants to identify lambda methods
  public static String walaLambdaStartString = "wala/lambda$";
  public static String lambdaMetafactoryClass = "java/lang/invoke/LambdaMetafactory";
  public static String walaArrayCopy = "com/ibm/wala/model/java/lang/System.arraycopy:(Ljava/lang/Object;Ljava/lang/Object;)V";
  public static String javaLibArrayCopy = "java/lang/System.arraycopy:(Ljava/lang/Object;ILjava/lang/Object;II)V";
  
  //Reformat the method if it is a lambda. Else simply return it.
  public static String reformatIfLambda(String inputMethod){
    String outputMethod;
    if (inputMethod.startsWith(walaLambdaStartString)){
      String fullLambdaSignature = inputMethod.substring(walaLambdaStartString.length()); //remove wala start string
      String lambdaSignatureWithoutArgs = fullLambdaSignature.split(":")[0];

      String classname = lambdaSignatureWithoutArgs.split("\\.")[0];
      String [] classnameListFormat = classname.split("\\$");
      String lambdaIndex = classnameListFormat[classnameListFormat.length-1];
      //remove the last element (the lambda index) from the classname
      classnameListFormat = Arrays.copyOf(classnameListFormat, classnameListFormat.length-1);
      String classnameFormatted = String.join("/",classnameListFormat);

      String methodname = lambdaSignatureWithoutArgs.split("\\.")[1];
      outputMethod = classnameFormatted + ".<lambda/" + methodname + "$" + lambdaIndex + ">:()V";
      return outputMethod;
    }
    else{ //If it is not a lambda method
      return inputMethod;
    }
  }

  //format the method to the required bytecode format
  public static String formatMethod(TypeName t,String methodname,Selector sel){
    String qualifiedClassName = "" + ( t.getPackage() == null ? "" : t.getPackage() + "/" ) + t.getClassName();
    if (qualifiedClassName.equals(lambdaMetafactoryClass)){
      return null; //don't want to use lambda metafactory nodes. They go nowhere, and don't appear in javaq
    }
    String formattedMethod = qualifiedClassName + "." + methodname + ":" + sel.getDescriptor();
    //Modify the method if it is a lambda
    formattedMethod = reformatIfLambda(formattedMethod);
    //If it is wala arrayCopy, replace with java Arraycopy
    if (formattedMethod.equals(walaArrayCopy)){
      formattedMethod = javaLibArrayCopy;
    }
    return formattedMethod;
  }

  //formats the final output line
  public static String formatFinalOutput(String firstMethod,String secondMethod,boolean bootSrcMethod,int off){
    //Decide the bytecode offset (and fix firstMethod) depending on if it is a boot method
    int bytecodeOffset;
    if (bootSrcMethod){
        firstMethod = "<boot>";
        bytecodeOffset = 0;
    } else {
        bytecodeOffset = off;
    }

    //Skip this edge if  destination node is a boot method
    if (secondMethod.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V")){
        return null;
    }   
    return firstMethod + "," + bytecodeOffset + "," + secondMethod + "\n";
  }

  
  public static void main(String[] args) throws WalaException, IllegalArgumentException, CancelException, IOException {
    Properties p = CommandLine.parse(args);
    String classpath = p.getProperty("classpath");
    String mainclass = p.getProperty("mainclass");
    String outputfile = p.getProperty("output");
    String resolveinterfaces = p.getProperty("resolveinterfaces");
    String exclude = p.getProperty("exclude");


    String analysis = p.getProperty("analysis");
    String reflection = p.getProperty("reflection"); 
    String handleStaticInit = p.getProperty("handleStaticInit");
    String useConstantSpecificKeys = p.getProperty("useConstantSpecificKeys");
    String handleZeroLengthArray = p.getProperty("handleZeroLengthArray");
    String useLexicalScopingForGlobals = p.getProperty("useLexicalScopingForGlobals");
    String useStacksForLexicalScoping = p.getProperty("useStacksForLexicalScoping");
   

    AnalysisScope scope = AnalysisScopeReader.instance.makeJavaBinaryAnalysisScope(classpath, null);
    ClassHierarchy cha = ClassHierarchyFactory.make(scope);

    Iterable<Entrypoint> entrypoints = Util.makeMainEntrypoints(cha, "L" + mainclass.replaceAll("\\.","/"));
    AnalysisOptions options = new AnalysisOptions(scope, entrypoints);

    /* Choose the reflection option */
    switch (reflection){
        case "NONE":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.NONE);
            break;
        case "FULL":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.FULL);
            break;
        case "APPLICATION_GET_METHOD":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.APPLICATION_GET_METHOD);
            break;
        case "NO_FLOW_TO_CASTS":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.NO_FLOW_TO_CASTS);
            break;
        case "NO_FLOW_TO_CASTS_APPLICATION_GET_METHOD":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.NO_FLOW_TO_CASTS_APPLICATION_GET_METHOD);
            break;
        case "NO_METHOD_INVOKE":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.NO_METHOD_INVOKE);
            break;
        case "NO_FLOW_TO_CASTS_NO_METHOD_INVOKE":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.NO_FLOW_TO_CASTS_NO_METHOD_INVOKE);
            break;
        case "ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE);
            break;
        case "ONE_FLOW_TO_CASTS_APPLICATION_GET_METHOD":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.ONE_FLOW_TO_CASTS_APPLICATION_GET_METHOD);
            break;
        case "MULTI_FLOW_TO_CASTS_APPLICATION_GET_METHOD":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.MULTI_FLOW_TO_CASTS_APPLICATION_GET_METHOD);
            break;
        case "NO_STRING_CONSTANTS":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.NO_STRING_CONSTANTS);
            break;
        case "STRING_ONLY":
            options.setReflectionOptions(AnalysisOptions.ReflectionOptions.STRING_ONLY);
            break;
    }

    // set other options

    if (handleStaticInit.equalsIgnoreCase("True")){
        options.setHandleStaticInit(true);
    } else {
        options.setHandleStaticInit(false);
    }
    if (useConstantSpecificKeys.equalsIgnoreCase("True")){
        options.setUseConstantSpecificKeys(true);
    } else {
        options.setUseConstantSpecificKeys(false);
    }
    if (handleZeroLengthArray.equalsIgnoreCase("True")){
        options.setHandleZeroLengthArray(true);
    } else {
        options.setHandleZeroLengthArray(false);
    }
    if (useLexicalScopingForGlobals.equalsIgnoreCase("True")){
        options.setUseLexicalScopingForGlobals(true);
    } else {
        options.setUseLexicalScopingForGlobals(false);
    }
    if (useStacksForLexicalScoping.equalsIgnoreCase("True")){
        options.setUseStacksForLexicalScoping(true);
    } else {
        options.setUseStacksForLexicalScoping(false);
    }

    /* Choose the correct analysis option */
    CallGraphBuilder builder;
    switch(analysis) {
        case "0cfa":
            builder = Util.makeZeroCFABuilder(Language.JAVA, options, new AnalysisCacheImpl(), cha);
            break;
        case "ZERO_CONTAINER_CFA":
            builder = Util.makeZeroContainerCFABuilder(options, new AnalysisCacheImpl(), cha);
            break;
        case "ZEROONE_CFA":
            builder = Util.makeZeroOneCFABuilder(Language.JAVA, options, new AnalysisCacheImpl(), cha);
            break;
        case "VANILLA_ZEROONECFA":
            builder = Util.makeVanillaZeroOneCFABuilder(Language.JAVA, options, new AnalysisCacheImpl(), cha);
            break;
        case "ZEROONE_CONTAINER_CFA":
            builder = Util.makeZeroOneContainerCFABuilder(options, new AnalysisCacheImpl(), cha);
            break;
        case "VANILLA_ZEROONE_CONTAINER_CFA":
            builder = Util.makeVanillaZeroOneContainerCFABuilder(options, new AnalysisCacheImpl(), cha);
            break;
        case "1cfa":
            builder = Util.makeNCFABuilder(1, options, new AnalysisCacheImpl(), cha);
            break;
        case "VANILLA_1cfa":
            builder = Util.makeVanillaNCFABuilder(1, options, new AnalysisCacheImpl(), cha);
            break;
        case "2cfa":
            builder = Util.makeNCFABuilder(2, options, new AnalysisCacheImpl(), cha);
            break;
        case "VANILLA_2cfa":
            builder = Util.makeVanillaNCFABuilder(2, options, new AnalysisCacheImpl(), cha);
            break;
        case "1obj":
            builder = Util.makeNObjBuilder(1, options, new AnalysisCacheImpl(), cha);
            break;
        case "VANILLA_1obj":
            builder = Util.makeVanillaNObjBuilder(1, options, new AnalysisCacheImpl(), cha);
            break;
        case "2obj":
            builder = Util.makeNObjBuilder(2, options, new AnalysisCacheImpl(), cha);
            break;
        case "VANILLA_2obj":
            builder = Util.makeVanillaNObjBuilder(2, options, new AnalysisCacheImpl(), cha);
            break;
        case "rta":
            builder = Util.makeRTABuilder(options, new AnalysisCacheImpl(), cha);
            break;
        default:
            System.out.println("-----Invalid analysis option----");
            builder = null;
    }
    
    CallGraph graph = builder.makeCallGraph(options, null);

    File file = new File(outputfile); 
    
    // Ensure parent directories exist
    file.getParentFile().mkdirs();
   
    file.createNewFile();
    FileWriter fw = new FileWriter(file);
    fw.write("method,offset,target\n"); //Header line
             
    for(Iterator<CGNode> it = graph.iterator(); it.hasNext(); ) {
        CGNode cgnode = it.next();
        IMethod m1 = cgnode.getMethod();
        TypeName t1 = m1.getDeclaringClass().getName();
        Selector sel1 = m1.getSelector();
        String name1 = sel1.getName().toString();
        String firstMethod = formatMethod(t1,name1,sel1);
        if (firstMethod==null){
          continue;
        }

        //Record if this is a fakeRoot/boot method or not
        boolean bootSrcMethod = (firstMethod.equals("com/ibm/wala/FakeRootClass.fakeRootMethod:()V") 
                        || firstMethod.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V"));

        for(Iterator<CallSiteReference> it2 =  cgnode.iterateCallSites(); it2.hasNext(); ) {
            CallSiteReference csref = it2.next();
            /* Choose to resolve the interface edge or not based on the input */
            if (resolveinterfaces.equalsIgnoreCase("true")){
                Set<CGNode> possibleActualTargets = graph.getPossibleTargets(cgnode,csref);
                for (CGNode cgnode2 : possibleActualTargets){
                    IMethod m2 = cgnode2.getMethod();
                    TypeName t2 = m2.getDeclaringClass().getName();
                    Selector sel2 = m2.getSelector();
                    String name2 = sel2.getName().toString();
                    String secondMethod = formatMethod(t2,name2,sel2);
                    if (secondMethod==null){
                      continue;
                    }
                    String formattedOutputLine =  formatFinalOutput(firstMethod,secondMethod,bootSrcMethod,csref.getProgramCounter());
                    if (formattedOutputLine!=null){
                      fw.write(formattedOutputLine); 
                    }
                }           
            } else {
                MethodReference m2 = csref.getDeclaredTarget();
                TypeName t2 = m2.getDeclaringClass().getName();
                Selector sel2 = m2.getSelector();
                String name2 = sel2.getName().toString();
                String secondMethod = formatMethod(t2,name2,sel2);
                if (secondMethod==null){
                      continue;
                }

                String formattedOutputLine =  formatFinalOutput(firstMethod,secondMethod,bootSrcMethod,csref.getProgramCounter());
                if (formattedOutputLine!=null){
                      fw.write(formattedOutputLine); 
                }
            }
        }
    }
    fw.close();  

  }
}