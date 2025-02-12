
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Properties;

import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.AnalysisScope;
import com.ibm.wala.ipa.cha.ClassHierarchy;
import com.ibm.wala.ipa.cha.ClassHierarchyFactory;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.WalaException;
import com.ibm.wala.util.config.AnalysisScopeReader;
import com.ibm.wala.util.io.CommandLine;


public class GetMethodCount { 
	public static void main(String[] args)
			throws WalaException, IllegalArgumentException, CancelException, IOException {
		// Read command-line args
		Properties p = CommandLine.parse(args);
		String classpath = p.getProperty("jarfile");
		String appClassesFile = p.getProperty("appClasses");
		String thirdPartylibClassesFile = p.getProperty("libClasses");
		
		// Read the application and third party classes
		HashSet<String> appClasses = readClasses(appClassesFile, true);
		HashSet<String> thirdPartyClasses = readClasses(thirdPartylibClassesFile, false);
		
		// Count the number of app and 3rd-party libs
		int appMethods = 0;
		int thirdPartyMethods = 0;
		AnalysisScope scope = AnalysisScopeReader.makeJavaBinaryAnalysisScope(classpath, null);
		ClassHierarchy cha = ClassHierarchyFactory.make(scope);
		
		for (IClass c : cha) {
			String classname = c.getName().toString();
			if (appClasses.contains(classname)) {
				appMethods += c.getDeclaredMethods().size();	
			} else if (thirdPartyClasses.contains(classname)) {
				thirdPartyMethods += c.getDeclaredMethods().size();
			}
		}
		System.out.println(appMethods);
		System.out.println(thirdPartyMethods);
	}
	
	
	
	// Format the name so that it matches the format used by Java
	private static String formatAppClassName(String classname) {
		String c = classname.replace('.', '/');
		return "L" + c;
	}
	
	private static HashSet<String> readClasses(String classesFile, boolean isAppClassesFile){
		HashSet<String> classes = new HashSet<String>();
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(classesFile));
			String line = reader.readLine();
			while (line != null) {
				if (isAppClassesFile) {
					classes.add(formatAppClassName(line));
				} else {
					classes.add(formatLibClassName(line));
				}
				line = reader.readLine();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return classes;
		
	}

	private static String formatLibClassName(String line) {
		// delete first two characters
		line = line.substring(2);
		// delete last 6 characters
		line = line.substring(0, line.length() - 6);
		// add the L
		line = "L" + line;
		return line;
	}
}

