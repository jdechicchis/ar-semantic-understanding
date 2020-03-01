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
    private Mesh mesh;
    
    private PixelInfo[] pixelLocations = new PixelInfo[0];
    private bool maskRequestMade = false;
    private bool randomPoints = false;

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
    }

    // Start is called before the first frame update
    void Start()
    {
        this.mesh = new Mesh();
        this.GetComponent<MeshFilter>().mesh = this.mesh;

        MLInput.Start();
    }

    // Update is called once per frame
    void Update()
    {
        this.getMask();

        if (this.randomPoints)
        {
            this.updateRandomMesh();
        }
        else
        {
            this.updateSemanticMesh();
        }
    }

    public void OnDestroy()
    {
        MLInput.OnControllerButtonDown -= OnButtonDown;
        MLInput.Stop();
    }

    void OnButtonDown(byte controller_id, MLInputControllerButton button) {
        if ((button == MLInputControllerButton.Bumper)) {
            this.randomPoints = !randomPoints;
        }
    }

    void updateRandomMesh()
    {
        int numPoints = 10000;

        Vector3[] points = new Vector3[numPoints];
        int[] indices = new int[numPoints];
        Color[] colors = new Color[numPoints];
        
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

        this.mesh.Clear();

        mesh.vertices = points;
        mesh.colors = colors;
        mesh.SetIndices(indices, MeshTopology.Points, 0);
    }

    private void updateSemanticMesh()
    {
        int numPoints = this.pixelLocations.Length;

        Vector3[] points = new Vector3[numPoints];
        int[] indices = new int[numPoints];
        Color[] colors = new Color[numPoints];
        
        int index = 0;
        foreach (PixelInfo pixelInfo in this.pixelLocations)
        {
            indices[index] = index;

            float z = (pixelInfo.z - 0.37f) / 1.5f;

            float x_width = this.horizontalWidthForZ(z);
            float y_width = this.verticalWidthForZ(z);

            float x_normalize = 224.0f / x_width;
            float y_normalize = 224.0f / y_width;

            float x = (pixelInfo.x - 100.0f) / x_normalize;
            float y = (pixelInfo.y - 100.0f) / y_normalize;

            points[index] = new Vector3(x, -y, z);
            colors[index] = this.classColors[pixelInfo.pixelClass - 1];

            index++;
        }
        
        this.mesh.Clear();

        mesh.vertices = points;
        mesh.colors = colors;
        mesh.SetIndices(indices, MeshTopology.Points, 0);
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

    private float horizontalWidthForZ(float z)
    {
        return 0.725f * z + 0.0649f;
    }

    private float verticalWidthForZ(float z)
    {
        return 0.521f * z + 0.0426f;
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
