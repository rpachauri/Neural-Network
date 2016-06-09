/**
 * This class keeps track of all sorts of patterns we may want to use to test XOROptimized
 *    
 * 
 * @author RyanPachauri
 * @version November 11, 2014
 */
public class InputPatterns
{
   /*
    * Array of NamedArray objects
    *    Each NamedArray object will keep track of
    *       a pattern and
    *       the name of that pattern
    *       the outputs when the pattern is inputed into the perceptron
    * PATTERNS are the patterns we train to
    * TEST_PATTERNS are the patterns we evaluate the network to once done training
    */
   public final static NamedArray[] PATTERNS =
      {InputPatterns.createBox(),
       InputPatterns.createCross()};
   public final static NamedArray[] TEST_PATTERNS =
      {InputPatterns.createBox(),                     //test the network against what we've trained
      InputPatterns.createCross(),                    //test the network against what we've trained
      InputPatterns.createBoxCross()};                //test the network against a combination of what we've trained
   
   /**
    * Creates the NamedArray for a box
    *    The name of the pattern is "box"
    *    The outputs of the pattern are {1,0}
    *       1 means that the pattern is a box
    *       0 means that the pattern is not a cross
    * @return NamedArray the pattern for a box
    */
   public static NamedArray createBox()
   {
      double[] box = {1,1,1,1,1,
                      1,0,0,0,1,
                      1,0,0,0,1,
                      1,0,0,0,1,
                      1,1,1,1,1};
      double[] outputs = {1,0};
      return new NamedArray("box", box, outputs);
   }//public NamedArray createBox()
   
   /**
    * Creates the NamedArray for a cross
    *    The name of the pattern is "cross"
    *    The outputs of the pattern are {0,1}
    *       0 means that the pattern is not a box
    *       1 means that the pattern is a cross
    * @return NamedArray the pattern for a cross
    */
   public static NamedArray createCross()
   {
      double[] cross = {1,0,0,0,1,
                        0,1,0,1,0,
                        0,0,1,0,0,
                        0,1,0,1,0,
                        1,0,0,0,1};
      double[] outputs = {0,1};
      return new NamedArray("cross", cross, outputs);
   }//public NamedArray createCross()
   
   /**
    * Creates the NamedArray for a boxcross
    *    The name of the pattern is "boxcross"
    *    The outputs of the pattern are {1,1}
    *       First 1 means that the pattern is a box
    *       Second 1 means that the pattern is a cross
    * @return NamedArray the pattern for a boxcross
    */
   public static NamedArray createBoxCross()
   {
      double[] boxcross = {1,1,1,1,1,
                           1,1,0,1,1,
                           1,0,1,0,1,
                           1,1,0,1,1,
                           1,1,1,1,1};
      double[] outputs = {1,1};
      return new NamedArray("boxcross", boxcross, outputs);
   }//public NamedArray createBoxCross()
}//public class InputPatterns
