/**
 * This class keeps track of three things that are meant to be together:
 *    String name
 *    double[] inputs
 *    double[] expectedOutputs
 * 
 * There is no default constructor because it would not make sense to have one.
 *    The class keeps track of something that the user wants to keep track of
 *    
 * @author RyanPachauri
 */
public class NamedArray
{
   private String name;//the name associated with the input
   private double[] inputs;//the array of inputs
   private double[] expectedOutputs;//the expected outputs when
   //the inputs are put in the perceptron
   
   /**
    * Constructor for the class NamedArray
    * 
    * There is no default constructor because it would not make sense to have one.
    *    The class keeps track of something that the user wants to keep track of
    * 
    * @param myName  String the name associated with the array
    * @param myInputs   the double[] array of inputs
    * @param myExpectedOutputs   the double[] array of expected outputs when
    *                            the inputs are put in the perceptron
    */
   public NamedArray(String myName, double[] myInputs, double[] myExpectedOutputs)
   {
      this.name = myName;
      this.inputs = myInputs;
      this.expectedOutputs = myExpectedOutputs;
   }//public NamedArray(String myName, double[] myInputs, double[] myExpectedOutputs)
   
   /**
    * Getter for the private instance variable String name
    * 
    * @return  name the private instance variable
    */
   public String getName()
   {
      return this.name;
   }//public String getName()
   
   /**
    * Getter for the private instance variable double[] inputs
    * 
    * @return inputs the private instance variable
    */
   public double[] getInputs()
   {
      return this.inputs;
   }//public double[] getInputs()
   
   /**
    * Getter for the private instance variable double[] expectedOutputs
    * 
    * @return expectedOutputs the private instance variable
    */
   public double[] getExpectedOutputs()
   {
      return this.expectedOutputs;
   }//public double[] getExpectedOutputs()
}//public class NamedArray
