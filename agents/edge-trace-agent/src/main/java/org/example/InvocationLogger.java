package org.example;

import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.io.FileWriter;
import java.io.File;
import java.io.IOException;
import java.lang.Object;
import java.util.regex.Matcher;
import java.util.regex.Pattern;



public class InvocationLogger {

    // edges to exclude
    private static ArrayList<ArrayList<Long>> excludeEdges = new ArrayList<>();
    // an arraylist of tuples (start, end)
    private static ArrayList<ArrayList<Long>> completeEdges = new ArrayList<>();
    
    
    private static HashMap<String, Long> instruction2start = new HashMap<>();
    private static ArrayList<String> trace = new ArrayList<>();
    private static long lineCounter = -1;
    private static int edgeCounter = 0;


    private static String reformatNodeString(String nodeString) {
        String pattern = "Node: < (?:Primordial|Application), ([^,]+), ([^ ]+) >";
        Pattern regex = Pattern.compile(pattern);
        Matcher matcher = regex.matcher(nodeString);

        if (matcher.find()) {
            String className = matcher.group(1);
            String methodSignature = matcher.group(2);

            // Remove leading 'L' if present
            if (className.startsWith("L")) {
                className = className.substring(1);
            }

            // Split method signature at '('
            int parenIndex = methodSignature.indexOf('(');
            if (parenIndex != -1) {
                String methodName = methodSignature.substring(0, parenIndex);
                String params = methodSignature.substring(parenIndex + 1);
                return className + "." + methodName + ":(" + params;
            }
        }

        return null; // Format incorrect
    }



    public static void addInstruction(Object instruction) {

        String instructionStr = instruction.toString();
        if (lineCounter == -1) {
            lineCounter = 0;
        }

        instruction2start.put(instructionStr, lineCounter);
    }

    public static void writeTrace(Object instruction, Object src, Object target, String programPath) {

        String instructionStr = instruction.toString();

        // getting the program counter from the instruction.
        Pattern pattern = Pattern.compile("@(\\d+)");
        Matcher matcher = pattern.matcher(instructionStr);
        int offset = -1;

        if (matcher.find()) {
            offset = Integer.parseInt(matcher.group(1));
        }
        else {
            return; // No program counter found
        }
        

        // find the start of the instruction, if exists, depending if the instruction is for jdk or not put it in completeedges or exclude edges
        if (instruction2start.containsKey(instructionStr)) {
            long start = instruction2start.get(instructionStr);
            

            ArrayList<Long> edge = new ArrayList<>();
            edge.add(start);
            edge.add(lineCounter);

            if (instructionStr.contains(", Ljava/lang")) {
                excludeEdges.add(edge);
            } else {

                // if there are any edges in the middle of the current edge, exclude them from the boundary of current edge and write the rest of the trace of current edge in the file
                ArrayList<String> filteredTrace = new ArrayList<>();
                
                for (long i = start; i < lineCounter; i++) {
                    // check all the boundaries of complete edges and exclude edges if there are between start and lineCounter and don't put those lines in the filteredTrace

                    boolean isExcluded = false;
                    for (ArrayList<Long> excludeEdge : excludeEdges) {
                        if (i >= excludeEdge.get(0) && i <= excludeEdge.get(1) && excludeEdge.get(0) > start) {
                            isExcluded = true;
                            break;
                        }
                    }
                    if (!isExcluded) {
                        for (ArrayList<Long> completeEdge : completeEdges) {
                            if (i >= completeEdge.get(0) && i <= completeEdge.get(1) && completeEdge.get(0) > start) {
                                isExcluded = true;
                                break;
                            }
                        }
                    }
                    if (!isExcluded) {
                        filteredTrace.add(trace.get((int) i));
                    }   
                }

                // write each element of the trace in a line to the file in the program dir path/edgeCounter.txt
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(programPath + "/" + edgeCounter++ + ".txt", true))) {
                    writer.write("Edge: " + src + "," + offset + "," + target);
                    writer.newLine();
                    for (String line : filteredTrace) {
                        writer.write(line);
                        writer.newLine();
                    }
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
                
                String srcStr = src.toString();
                String targetStr = target.toString();
                if (srcStr.contains("fakeRootMethod") || srcStr.contains("fakeWorldClinit")) {
                    srcStr = "<boot>";
                    offset = 0;
                }
                else{
                    // reformat the src and target string to the required format
                    srcStr = reformatNodeString(srcStr);
                }
                targetStr = reformatNodeString(targetStr);
                if (srcStr == null || targetStr == null) {
                    return; // Format incorrect
                }


                // append the srcStr, target to a csv file with the correponding edge counter
                File file = new File(programPath + "/edges.csv");
                boolean fileExists = file.exists();

                try (BufferedWriter writer = new BufferedWriter(new FileWriter(file, true))) {
                    // Write header if the file is newly created
                    if (!fileExists) {
                        writer.write("edge_id,method,offset,target");
                        writer.newLine();
                    }

                    // Append the edge row
                    writer.write((edgeCounter - 1) + "," + srcStr + "," + offset + "," + targetStr);
                    writer.newLine();
                } catch (IOException e) {
                    e.printStackTrace();
}

                completeEdges.add(edge);
            }

        }
        
    }

    public static void addLineToTrace(String className, String methodName, String desc, String owner, String name, String descriptor) {
        
        // add line to the trace, increament the line counter
        if (lineCounter != -1) {
            trace.add(className + "." + methodName + " " + desc + "," + owner + "." + name + " " + descriptor);
            lineCounter++;  
        }    
    }

    public static void addLineToTrace(String className, String methodName, String desc,  int ifCounter, boolean isIFBranch) {
        // add line to the trace, increament the line counter
        if (lineCounter != -1) {

            String ifStatementId = className + "." + methodName + " " + desc;
            if (isIFBranch)
                ifStatementId +=  ":IF#" + ifCounter;
            else
                ifStatementId +=  ":ELSE#" + ifCounter;

            trace.add(ifStatementId);
            lineCounter++;  
        }    
    }
}
