using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.MagicLeap;

public class CanvasControl : MonoBehaviour
{
    public GameObject Pixel;
    public Camera Camera;
    public Text TextBox;
    public Material ClassOneMaterial;
    public Material ClassTwoMaterial;
    public Material ClassThreeMaterial;
    public Material ClassFourMaterial;
    public Material ClassFiveMaterial;
    public Material ClassSixMaterial;
    public Material ClassSevenMaterial;

    private Mesh mesh;

    private List<GameObject> pixels;
    private MLInputController controller;
    private float triggerThreshold = 0.2f;

    // Pixel width/height/depth
    private float pixelSize = 0.01f;

    private int numFrames = 0;

    private void Awake()
    {
        //this.controller = MLInput.GetController(MLInput.Hand.Right);
        /*
        MLInputConfiguration config = new MLInputConfiguration(MLInputConfiguration.DEFAULT_TRIGGER_DOWN_THRESHOLD,
        MLInputConfiguration.DEFAULT_TRIGGER_UP_THRESHOLD,
        true);


        MLResult result = MLInput.Start(config);
        if (!result.IsOk)
        {
            Debug.LogErrorFormat("Error: ControllerConnectionHandler failed starting MLInput, disabling script. Reason: {0}", result);
            //enabled = false;
            return;
        }

        MLInputController m_leftController = MLInput.GetController(MLInput.Hand.Left);
        MLInputController m_rightController = MLInput.GetController(MLInput.Hand.Right);
        */

        //MLInput.OnControllerButtonDown += HandleOnButtonDown;
        //MLInput.OnControllerButtonUp += HandleOnButtonUp;
    }

    // Start is called before the first frame update
    void Start()
    {
        this.pixels = new List<GameObject>();
        //this.mesh = new Mesh();
        //GetComponent<MeshFilter>().mesh = mesh;
    }

    // Update is called once per frame
    void Update()
    {
        if (numFrames < 5) {
            numFrames++;
            return;
        }

        numFrames = 0;

        //this.TextBox.transform.localPosition = new Vector3(0.0f, 0.0f, 2.0f);
        //this.TextBox.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);

        Debug.Log("Here!");

        //return;

        //if (this.controller.TriggerValue <= this.triggerThreshold) return;


        /*
        this.mesh.Clear();

        int numPoints = 100;

        Vector3[] points = new Vector3[numPoints];
        int[] indecies = new int[numPoints];
        Color[] colors = new Color[numPoints];
        for(int i=0;i<points.Length;++i) {
            points[i] = new Vector3(UnityEngine.Random.Range(-10,10), UnityEngine.Random.Range (-10,10), UnityEngine.Random.Range (-10,10));
            indecies[i] = i;
            colors[i] = new Color(UnityEngine.Random.Range(0.0f,1.0f),UnityEngine.Random.Range (0.0f,1.0f),UnityEngine.Random.Range(0.0f,1.0f),1.0f);
        }

        mesh.vertices = points;
        mesh.colors = colors;
        mesh.SetIndices(indecies, MeshTopology.Points,0);
        */
        return;

        foreach (GameObject pixel in this.pixels) {
            Destroy(pixel);
        }

        this.pixels.Clear();

        PixelInfo[] pixelLocations = this.getMask();

        Debug.Log("Updates");

        Debug.Log(string.Format("Num labels: {0}", pixelLocations.Length));

        int count = 0;
        foreach (PixelInfo pixelInfo in pixelLocations) {
            if (count % 10 == 0) {
                
            //if (pixelInfo.z > 0.5f) continue;
            GameObject pixel = Instantiate(this.Pixel);
            pixel.transform.SetParent(this.transform);
            float x = ((pixelInfo.x - 122) / 224.0f - 0.0f) * 1.5f;
            float y = ((pixelInfo.y - 122) / 224.0f + 0.5f) * 1.5f;
            float z = (pixelInfo.z + 1.0f);//0.075
            //pixel.transform.localPosition = this.Camera.ViewportToWorldPoint(new Vector3(0, 0, 2));
            //pixel.transform.localPosition = new Vector3(0, 0, 1);
            pixel.transform.localPosition = new Vector3(x, -y, z);
            pixel.transform.localScale = new Vector3(this.pixelSize, this.pixelSize, this.pixelSize);

            MeshRenderer pixelRenderer = pixel.GetComponent<MeshRenderer>();
            switch (pixelInfo.pixelClass) {
                case 1:
                    pixelRenderer.material = this.ClassOneMaterial;
                    break;
                case 2:
                    pixelRenderer.material = this.ClassTwoMaterial;
                    break;
                case 3:
                    pixelRenderer.material = this.ClassThreeMaterial;
                    break;
                case 4:
                    pixelRenderer.material = this.ClassFourMaterial;
                    break;
                case 5:
                    pixelRenderer.material = this.ClassFiveMaterial;
                    break;
                case 6:
                    pixelRenderer.material = this.ClassSixMaterial;
                    break;
                case 7:
                    pixelRenderer.material = this.ClassSevenMaterial;
                    break;
                default:
                    Debug.Log("INVALID CLASS!!");
                    break;
            }

            this.pixels.Add(pixel);
            }
            count++;
        }
    }

    private PixelInfo[] getMask()
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://10.197.154.206:5005/mask");
        HttpWebResponse response = (HttpWebResponse)request.GetResponse();
        StreamReader reader = new StreamReader(response.GetResponseStream());
        string jsonResponse = reader.ReadToEnd();
        LocationData locationData = JsonUtility.FromJson<LocationData>(jsonResponse);
          
        return locationData.locations;
    }
}

[Serializable]
public class LocationData
{
    public PixelInfo[] locations;
}

[Serializable]
public class PixelInfo {
    public float x;
    public float y;
    public float z;
    public int pixelClass;

    public PixelInfo(float x, float y, float z, int pixelClass)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.pixelClass = pixelClass;
    }
}