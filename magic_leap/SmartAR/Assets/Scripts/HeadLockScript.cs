using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeadLockScript : MonoBehaviour
{
    public GameObject Camera;
    private float distance = 1.0f;
    
    public void HardHeadLock(GameObject obj) {
        obj.transform.position = this.Camera.transform.position + this.Camera.transform.forward * this.distance;
        obj.transform.rotation = this.Camera.transform.rotation;
    }
}
