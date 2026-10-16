import uuid


def test_create_and_list_own_projects(api, signup):
    h = signup()
    r = api.post("/api/v1/projects", headers=h, json={"name": "  Sepsis review  "})
    assert r.status_code == 201 and r.json()["name"] == "Sepsis review"
    assert [p["name"] for p in api.get("/api/v1/projects", headers=h).json()] == ["Sepsis review"]


def test_blank_or_long_names_rejected(api, signup):
    h = signup()
    for name in ("", "   ", "x" * 201):
        assert api.post("/api/v1/projects", headers=h, json={"name": name}).status_code == 422


def test_projects_are_private(api, signup):
    alice, bob = signup("alice@example.com"), signup("bob@example.com")
    pid = api.post("/api/v1/projects", headers=alice, json={"name": "A"}).json()["id"]
    assert api.get("/api/v1/projects", headers=bob).json() == []
    assert api.get(f"/api/v1/projects/{pid}/documents", headers=bob).status_code == 404
    assert api.get(f"/api/v1/projects/{pid}/documents", headers=alice).status_code == 200
    assert api.get(f"/api/v1/projects/{uuid.uuid4()}/documents", headers=alice).status_code == 404
    assert api.get("/api/v1/projects/not-a-uuid/documents", headers=alice).status_code == 422
