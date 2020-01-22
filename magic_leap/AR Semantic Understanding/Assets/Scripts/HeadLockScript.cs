using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeadLockScript : MonoBehaviour
{
    public GameObject Camera;

    private const float _distance = 0.1f;

    public void HardHeadLock(GameObject obj)
    {
        obj.transform.position = Camera.transform.position + Camera.transform.forward * _distance;
        obj.transform.rotation = Camera.transform.rotation;
    }

    public void HeadLock(GameObject obj, float speed)
    {
        speed = Time.deltaTime * speed;
        Vector3 posTo = Camera.transform.position + Camera.transform.forward * _distance;
        obj.transform.position = Vector3.SlerpUnclamped(obj.transform.position, posTo, speed);
        Quaternion rotTo = Quaternion.LookRotation(obj.transform.position - Camera.transform.position);
        obj.transform.rotation = Quaternion.Slerp(obj.transform.rotation, rotTo, speed);
    }
}