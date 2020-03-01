using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.MagicLeap;

public class MeshGenerator : MonoBehaviour
{
    public Text Text;
    private Mesh mesh;
    
    private PixelInfo[] pixelLocations = new PixelInfo[0];
    private bool maskRequestMade = false;

    private float z = 0.0f;
    private float x_width = 0.001f;
    private float y_width = 0.001f;
    private MLInputController controller;
    private DateTime lastTriggerTime = DateTime.Now;
    private int change_dir = 0;
    private bool change_increase = true;

    // Color of class labels
    // Class 0 is omitted so index with (class label - 1)
    private Color [] classColors = new Color[]
    {
        new Color(0.0f, 1.0f, 1.0f, 1.0f),
        new Color(1.0f, 0.0f, 1.0f, 1.0f),
        new Color(1.0f, 1.0f, 0.0f, 1.0f),
        new Color(1.0f, 0.0f, 0.0f, 1.0f),
        new Color(0.0f, 1.0f, 0.0f, 1.0f),
        new Color(0.0f, 0.0f, 1.0f, 1.0f),
        new Color(0.8f, 0.4f, 0.0f, 1.0f)
    };

    void Awake()
    {
        MLInput.OnControllerButtonDown += OnButtonDown;
        MLInput.OnControllerButtonUp += OnButtonUp;
    }

    // Start is called before the first frame update
    void Start()
    {
        this.mesh = new Mesh();
        this.GetComponent<MeshFilter>().mesh = this.mesh;

        MLInput.Start();

        this.controller = MLInput.GetController(MLInput.Hand.Left);
    }

    // Update is called once per frame
    void Update()
    {
        UpdateMesh();

        if (this.controller.TriggerValue > 0.2f && (DateTime.Now - this.lastTriggerTime).TotalSeconds > 0.5) {
            Debug.Log("Trigger tap");
            this.lastTriggerTime = DateTime.Now;
            this.change_dir += 1;
            if (this.change_dir > 2)
            {
                this.change_dir = 0;
            }
        }
    }

    public void OnDestroy()
    {
        MLInput.OnControllerButtonDown -= OnButtonDown;
        MLInput.OnControllerButtonUp -= OnButtonUp;
        MLInput.Stop();
    }

    /*
    void CreateMesh()
    {
        Vector3[] vertices = new Vector3[]
        {
            //this.Camera.ViewportToWorldPoint(new Vector3(0.5f, 0.5f, 0.5f))
            //new Vector3(0.0f, -0.15f, 0.5f)
            //new Vector3(0, 0.05f, 0),
            //new Vector3(0.05f, 0, 0)
        };

        this.triangles = new int[]
        {
            0//, 1, 2
        };
    }
    */

    void OnButtonDown(byte controller_id, MLInputControllerButton button) {
        if ((button == MLInputControllerButton.Bumper)) {
            Debug.Log("Bumper tap");
            if (this.change_increase)
            {
                if (this.change_dir == 0)
                {
                    this.x_width += 0.001f;
                }
                else if (this.change_dir == 1)
                {
                    this.y_width += 0.001f;
                }
                else if (this.change_dir == 2)
                {
                    this.z += 0.1f;
                }
            }
            else
            {
                if (this.change_dir == 0)
                {
                    this.x_width -= 0.001f;
                }
                else if (this.change_dir == 1)
                {
                    this.y_width -= 0.001f;
                }
                else if (this.change_dir == 2)
                {
                    this.z -= 0.1f;
                }
            }
        }
    }

    void OnButtonUp(byte controller_id, MLInputControllerButton button)
    {
        if (button == MLInputControllerButton.HomeTap)
        {
            Debug.Log("Home tap");
            this.change_increase = !this.change_increase;
        }
    }

    void UpdateMesh ()
    {
        // 0.370 is min ML display distance in z
        // canvas z offset to 0.4
        // x from -0.2 to 0.2 at z of 0.5
        // y from -0.15 to 0.15 at z of 0.5
        this.mesh.Clear();

        Vector3[] vertices = new Vector3[]
        {
            new Vector3(0.0f, 0.0f, this.z),  // mid point
            new Vector3(-this.x_width/2.0f, 0.0f, this.z),  // left point
            new Vector3(this.x_width/2.0f, 0.0f, this.z),  // right point
            new Vector3(0.0f, this.y_width/2.0f, this.z),  // top point
            new Vector3(0.0f, -this.y_width/2.0f, this.z), // bottom point
        };

        Color[] colors = new Color[]
        {
            new Color(1.0f, 0.0f, 0.0f, 1.0f),
            new Color(0.0f, 1.0f, 0.0f, 1.0f),
            new Color(0.0f, 1.0f, 0.0f, 1.0f),
            new Color(1.0f, 0.0f, 1.0f, 1.0f),
            new Color(1.0f, 0.0f, 1.0f, 1.0f)
        };

        this.mesh.vertices = vertices;
        this.mesh.colors = colors;
        //this.mesh.triangles = triangles;

        int[] indices = new int[]
        {
            0, 1, 2, 3, 4
        };

        this.mesh.SetIndices(indices, MeshTopology.Points, 0);
        
        string change_string = "";
        if (this.change_dir == 0)
        {
            change_string = "x";
        }
        else if (this.change_dir == 1)
        {
            change_string = "y";
        }
        else if (this.change_dir == 2)
        {
            change_string = "z";
        }

        this.Text.text = string.Format("x: {0}\ny: {1}\nz: {2}\nchange {3} ({4})",
                                       this.x_width,
                                       this.y_width,
                                       this.z,
                                       change_string,
                                       this.change_increase ? "+" : "-");

        return;
        /*
        this.mesh.Clear();

        //PixelInfo[] pixelLocations = this.getMask();

        //Debug.Log(string.Format("Num points: {0}", pixelLocations.Length));

        this.getMask();

        int numPoints = this.pixelLocations.Length;

        Vector3[] points = new Vector3[numPoints];
        int[] indices = new int[numPoints];
        Color[] colors = new Color[numPoints];
        */
        /*
        for(int i=0; i<points.Length; i++) {
            points[i] = new Vector3(
                UnityEngine.Random.Range(-0.2f, 0.2f),
                UnityEngine.Random.Range(-0.15f, 0.15f),
                UnityEngine.Random.Range (0.0f, 5.0f));
            indices[i] = i;
            colors[i] = new Color(
                UnityEngine.Random.Range(0.0f, 1.0f),
                UnityEngine.Random.Range(0.0f, 1.0f),
                UnityEngine.Random.Range(0.0f, 1.0f),
                1.0f);
        }
        */

        // 0.370 is min ML display distance in z
        // canvas z offset to 0.4
        // x from -0.2 to 0.2 at z of 0.5
        // y from -0.15 to 0.15 at z of 0.5

        /*
        int index = 0;
        foreach (PixelInfo pixelInfo in this.pixelLocations)
        {
            indices[index] = index;

            float x = ((pixelInfo.x / 560.0f) - 0.2f);
            float y = ((pixelInfo.y / 747.0f) - 0.15f);
            float z = (pixelInfo.z - 0.37f) / 2.0f - 0.1f;

            points[index] = new Vector3(x, -y, z);
            colors[index] = this.classColors[pixelInfo.pixelClass - 1];

            index++;
        }

        mesh.vertices = points;
        mesh.colors = colors;
        mesh.SetIndices(indices, MeshTopology.Points, 0);
        */
    }
    
    private async void getMask()
    {
        if (this.maskRequestMade) return;

        try
        {
            this.maskRequestMade = true;
            Task.Run(() => { makeMaskRequest(); });
        }
        catch (Exception)
        {
            this.maskRequestMade = false;
        }
    }

    public async Task makeMaskRequest()
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://192.168.1.18:5005/mask");
        HttpWebResponse response = (HttpWebResponse)request.GetResponse();
        StreamReader reader = new StreamReader(response.GetResponseStream());
        string jsonResponse = reader.ReadToEnd();
        LocationData locationData = JsonUtility.FromJson<LocationData>(jsonResponse);
        
        this.pixelLocations = locationData.locations;

        this.maskRequestMade = false;
    }

    /*
    private PixelInfo[] getMask()
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://192.168.1.18:5005/mask");
        HttpWebResponse response = (HttpWebResponse)request.GetResponse();
        StreamReader reader = new StreamReader(response.GetResponseStream());
        string jsonResponse = reader.ReadToEnd();
        LocationData locationData = JsonUtility.FromJson<LocationData>(jsonResponse);
          
        return locationData.locations;
    }
    */
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
