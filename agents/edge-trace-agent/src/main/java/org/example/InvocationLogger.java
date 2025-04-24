package org.example;

import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Object;



// public class InvocationLogger {
//     private static ArrayList<String> activeInstructions = new ArrayList<>();
//     private static HashMap<String, ArrayList<String>> instruction2trace = new HashMap<>();
//     private static int edgeCounter = 0;

//     public static void addInstruction(Object instruction) {
//         String instructionStr = instruction.toString();

//         if (instructionStr.contains(", Ljava/lang")){
//             return;
//         }
//         activeInstructions.add(instructionStr);
//         ArrayList<String> trace = new ArrayList<>();
//         instruction2trace.put(instructionStr, trace);
//     }

//     public static void writeTrace(Object instruction, Object src, Object target, String programPath) {

//         System.out.println("Writing trace for instruction: " + instruction);

//         // convert the instruction to string
//         String instructionStr = instruction.toString();
        
//         // if instruction exists in the map, write the current trace of it to a file
//         if (activeInstructions.contains(instructionStr)) {
//             ArrayList<String> trace = instruction2trace.get(instructionStr);
            
//             // write each element of the trace in a line to the file in the program dir path/edgeCounter.txt
//             try (BufferedWriter writer = new BufferedWriter(new FileWriter(programPath + "/" + edgeCounter++ + ".txt", true))) {
//                 writer.write("Edge: " + src + "," + target);
//                 writer.newLine();
//                 for (String edge : trace) {
//                     writer.write(edge);
//                     writer.newLine();
//                 }
//             }
//             catch (IOException e) {
//                 e.printStackTrace();
//             }
//         }

//         // if instruction is a staticinovke, remove instruction from the map
//         if (instructionStr.contains("INVOKESTATIC")) {
//             activeInstructions.remove(instructionStr);
//         }
//     }

//     public static void addLineToTrace(String className, String methodName, String desc, String owner, String name, String descriptor) {
//         // add the element to all of the traces of all the active instructions
        

//         for (String instruction : activeInstructions) {
//             ArrayList<String> trace = instruction2trace.get(instruction);
//             trace.add(className + "." + methodName + " " + desc + "," + owner + "." + name + " " + descriptor);
//         }
//     }
// }


public class InvocationLogger {

    // edges to exclude
    private static ArrayList<ArrayList<Long>> excludeEdges = new ArrayList<>();
    // an arraylist of tuples (start, end)
    private static ArrayList<ArrayList<Long>> completeEdges = new ArrayList<>();
    
    
    private static HashMap<String, Long> instruction2start = new HashMap<>();
    private static ArrayList<String> trace = new ArrayList<>();
    private static long lineCounter = -1;
    
    
    private static int edgeCounter = 0;



    public static void addInstruction(Object instruction) {

        String instructionStr = instruction.toString();
        if (lineCounter == -1) {
            lineCounter = 0;
        }

        instruction2start.put(instructionStr, lineCounter);
    }

    public static void writeTrace(Object instruction, Object src, Object target, String programPath) {

        String instructionStr = instruction.toString();

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
                    writer.write("Edge: " + src + "," + target);
                    writer.newLine();
                    for (String line : filteredTrace) {
                        writer.write(line);
                        writer.newLine();
                    }
                }
                catch (IOException e) {
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
}
